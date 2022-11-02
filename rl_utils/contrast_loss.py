import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._inner_cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _inner_cosine_similarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        #print(f"[tlog] zis: {zis.size()}")
        #print(f"[tlog] zjs: {zjs.size()}")
        batch_size = zis.size(0)
        
        if batch_size != self.batch_size: #reset mask by tzy for inconsistent batch size
            self.batch_size = batch_size
            self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
            
        representations = torch.cat([zjs, zis], dim=0)
        #print(f"[tlog] representations: {representations.size()}")
        
        similarity_matrix = self.similarity_function(representations, representations)
        #print(f"[tlog] similarity_matrix: {similarity_matrix.size()}") #[64, 64]
        #rint(f"[tlog] similarity_matrix: {similarity_matrix}") #[64, 64]
        #sys.exit(0)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        #print(f"[tlog] l_pos: {l_pos.size()}") #32
        #print(f"[tlog] l_pos: {l_pos}")
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        #print(f"[tlog] r_pos: {r_pos.size()}") #32
        #print(f"[tlog] r_pos: {r_pos}")
        
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) #64, 1
        #print(f"[tlog] positives: {positives.size()}")
        #print(f"[tlog] positives: {positives}")
        #sys.exit(0)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) #64, 62
        #print(f"[tlog] negatives: {negatives.size()}")
        #print(f"[tlog] negatives: {negatives}")
        #sys.exit(0)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)