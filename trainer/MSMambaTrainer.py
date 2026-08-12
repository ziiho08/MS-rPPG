"""Trainer for MSMamba."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.MsMamba import MsMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.RythmFormerLossComputer import Hybrid_Loss


class MSMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.seed = config.SEED
        self.diff_flag = 0
        self.data_dict = {}
        self.dataset = config.TRAIN.DATA.DATASET
        self.train_modality = getattr(config.TRAIN.DATA, 'MODALITY', 'MULTI_SPECTRAL').upper()
        self.valid_modality = getattr(config.VALID.DATA, 'MODALITY', 'MULTI_SPECTRAL').upper()
        self.test_modality  = getattr(config.TEST.DATA,  'MODALITY', 'MULTI_SPECTRAL').upper()

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1

        cfg_m = config.MODEL.MSMamba
        def _build_model():
            return MsMamba(
                frames=cfg_m.FRAMES,
                depth=cfg_m.DEPTH,
                embed_dim=cfg_m.EMBED_DIM,
                drop_path_rate=config.MODEL.DROP_RATE,
            ).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = _build_model()
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Hybrid_Loss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches
            )
        elif config.TOOLBOX_MODE == "only_test":
            self.model = _build_model()
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("MsMamba trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                if self.train_modality == 'MULTI_SPECTRAL':
                    rgb_batch, nir_batch = batch[0]
                    nir = nir_batch.float().to(self.device)
                elif self.train_modality == 'NIR':
                    rgb_batch = None
                    nir = batch[0].float().to(self.device)
                else:
                    rgb_batch = batch[0]
                    nir = None
                labels = batch[1].float()

                rgb = rgb_batch.float().to(self.device) if rgb_batch is not None else None
                labels = labels.to(self.device)
                N = (rgb if rgb is not None else nir).shape[0]

                self.optimizer.zero_grad()
                pred_ppg = self.model(rgb, nir)
                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize

                loss = 0.0
                for ib in range(N):
                    loss = loss + self.criterion(pred_ppg[ib], labels[ib], epoch , self.config.TRAIN.DATA.FS , self.diff_flag)
                loss = loss / N

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for _, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                if self.valid_modality == 'MULTI_SPECTRAL':
                    rgb_valid, nir_valid = valid_batch[0]
                elif self.valid_modality == 'NIR':
                    rgb_valid = None
                    nir_valid = valid_batch[0]
                else:
                    rgb_valid = valid_batch[0]
                    nir_valid = None
                labels_valid = valid_batch[1].to(self.config.DEVICE)

                rgb_valid = rgb_valid.float().to(self.device) if rgb_valid is not None else None
                if nir_valid is not None:
                    nir_valid = nir_valid.float().to(self.device)

                pred_ppg_valid = self.model(rgb_valid, nir_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '_Seed' + str(self.seed) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '_Seed' + str(self.seed) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        predictions = dict()
        labels = dict()
        for batch_idx, test_batch in enumerate(data_loader['test']):
            chunk_len = self.chunk_len

            if self.test_modality == 'MULTI_SPECTRAL':
                rgb_batch, nir_batch = test_batch[0]
            elif self.test_modality == 'NIR':
                rgb_batch = None
                nir_batch = test_batch[0]
            else:
                rgb_batch = test_batch[0]
                nir_batch = None
            labels_test = test_batch[1].to(self.config.DEVICE)

            rgb = rgb_batch.float().to(self.device) if rgb_batch is not None else None
            nir = nir_batch.float().to(self.device) if nir_batch is not None else None

            N = (rgb if rgb is not None else nir).shape[0]

            with torch.no_grad():
                pred_ppg_test = self.model(rgb, nir)
                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test, axis=-1).view(-1, 1)) / torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize

            labels_test = labels_test.view(-1, 1)
            pred_ppg_test = pred_ppg_test.view(-1, 1)
            for ib in range(N):
                subj_index = test_batch[2][ib]
                sort_index = int(test_batch[3][ib])
                if subj_index not in predictions.keys():
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()
                predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]

        print(' ')
        BaseTrainer.save_test_outputs(self, predictions, labels, self.config)
        calculate_metrics(predictions, labels, self.config)

        if self.config.TEST.DATA.DATASET.upper() == 'PHYSDRIVE':
            from evaluation.physdrive_conditions import calculate_metrics_by_condition
            calculate_metrics_by_condition(predictions, labels, self.config)
        elif self.config.TEST.DATA.DATASET.upper() == 'MR-NIRP':
            from evaluation.mrnirp_conditions import calculate_metrics_by_condition
            calculate_metrics_by_condition(predictions, labels, self.config)


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '_Seed' + str(self.seed) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
        if index > 0:
            prev_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(index - 1) + '_Seed' + str(self.seed) + '.pth')
            keep_prev = (not self.config.TEST.USE_LAST_EPOCH) and (index - 1 == self.best_epoch)
            if not keep_prev and os.path.exists(prev_path):
                os.remove(prev_path)


    def data_augmentation(self,data,labels,index1,index2):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        rand1_vals = np.random.random(N)
        rand2_vals = np.random.random(N)
        for idx in range(N):
            index = index1[idx] + index2[idx]
            rand1 = rand1_vals[idx]
            rand2 = rand2_vals[idx]
            if rand1 < 0.5 :
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                    
                if gt_hr_fft > 90: 
                    rand3 = random.randint(0, D//2-1)
                    even_indices = torch.arange(0, D, 2)
                    odd_indices = even_indices + 1
                    data_aug[:, even_indices, :, :, :] = data[:, rand3 + even_indices// 2, :, :, :]
                    labels_aug[:, even_indices] = labels[:, rand3 + even_indices // 2]
                    data_aug[:, odd_indices, :, :, :] = (data[:, rand3 + odd_indices // 2, :, :, :] + data[:, rand3 + (odd_indices // 2) + 1, :, :, :]) / 2
                    labels_aug[:, odd_indices] = (labels[:, rand3 + odd_indices // 2] + labels[:, rand3 + (odd_indices // 2) + 1]) / 2
                elif gt_hr_fft < 75 :
                    data_aug[:, :D//2, :, :, :] = data[:, ::2, :, :, :]
                    labels_aug[:, :D//2] = labels[:, ::2]
                    data_aug[:, D//2:, :, :, :] = data_aug[:, :D//2, :, :, :]
                    labels_aug[:, D//2:] = labels_aug[:, :D//2]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                      
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug