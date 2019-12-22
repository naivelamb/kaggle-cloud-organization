# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:34:25



from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from util.utils import *
from common import *
import glob
import os

class Dataset_cloud(Dataset):
    def __init__(self, root:Path, df: pd.DataFrame, non_emptiness=None, mode='train', transform=None, cls_label=None):
        super(Dataset_cloud, self).__init__()
        self._root = root
        self._df = df
        self._image_transform = transform
        self._mode = mode
        self.non_emptiness = non_emptiness
        self._cls_label = cls_label

    def __len__(self):
        return len(self._df)

    def load_image(self, filePath):
        image = cv2.imread(str(filePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_mask(self, rle, h=1400, w=2100):
        return rle2mask(rle, height=h, width=w)

    def __getitem__(self, index):
        item = self._df.iloc[index]
        name = item['Image_Label']
        img_path = self._root / f'{name}'
        img = self.load_image(img_path)
        h, w, _ = img.shape
        if self._mode == 'train':
            # generate rles for each channel
            if self._cls_label in [0,1,2,3]:
                mask = np.zeros((h, w, 1), dtype=np.float32)
                if str(self._cls_label) not in item['Labels']:
                    pass
                else:
                    labels = [int(x) for x in item['Labels'].split(' ')]
                    rles = item['EncodedPixels'].split('|')
                    for label, rle in zip(labels, rles):
                        if label == self._cls_label:
                            mask[:, :, 0] = self.load_mask(rle, h=h, w=w)
            else:
                mask = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
                if len(item['Labels']) == 1:
                    label = int(item['Labels'])
                    rle = item['EncodedPixels']
                    mask[:, :, label] = self.load_mask(rle, h=h, w=w)
                else:
                    labels = [int(x) for x in item['Labels'].split(' ')]
                    rles = item['EncodedPixels'].split('|')
                    for label, rle in zip(labels, rles):
                        mask[:, :, label] = self.load_mask(rle, h=h, w=w)

            if self._image_transform:
                augment = self._image_transform(image=img, mask=mask)
                img=augment['image']
                mask=augment['mask']
                mask=mask[0].permute(2, 0, 1)
            return img, mask
        elif self._mode == 'test':
            if self._image_transform:
                augment = self._image_transform(image=img)
                img = augment['image']
            if self._cls_label in [0,1,2,3]:
                names = ['%s_%s' % (name, CLASS_NAMES[self._cls_label])]
            else:
                names = ['%s_%s' % (name, CLASS_NAMES[x]) for x in range(4)]
            return img, names

class EmptySampler(Sampler):
    def __init__(self, data_source, positive_ratio_range: Tuple[float, float], epochs: int = 50):
        super().__init__(data_source)
        assert len(positive_ratio_range) == 2
        self.positive_indices = np.where(data_source.non_emptiness == 1)[0]
        self.negative_indices = np.where(data_source.non_emptiness == 0)[0]
        self.positive_ratio_range = positive_ratio_range
        self.positive_num: int = len(self.positive_indices)
        self.current_epoch: int = 0
        self.epochs: int = epochs

    @property
    def positive_ratio(self) -> float:
        np.random.seed(self.current_epoch)
        min_ratio, max_ratio = self.positive_ratio_range
        return max_ratio - (max_ratio - min_ratio) / self.epochs * self.current_epoch

    @property
    def negative_num(self) -> int:
        assert self.positive_ratio <= 1.0
        return int(self.positive_num // self.positive_ratio - self.positive_num)

    def __iter__(self):
        negative_indices = np.random.choice(self.negative_indices, size=self.negative_num)
        indices = np.random.permutation(np.hstack((negative_indices, self.positive_indices)))
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.positive_num + self.negative_num

    def set_epoch(self, epoch):
        self.current_epoch = epoch
