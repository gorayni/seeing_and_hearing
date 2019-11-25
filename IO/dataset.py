from __future__ import print_function, division
import torch.utils.data as data
import numpy as np
import tables
from numpy.random import randint
import pickle

class EpicTSNDataset(data.Dataset):
    def __init__(self,
                 epic_dataset,
                 classes_map,
                 splits,
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 transform=None,
                 random_shift=True,
                 split_name='train',
                 classification_type='verb',
                 part=-1
                 ):
        self.epic_dataset = epic_dataset
        self.classes_map = classes_map
        self.splits = splits
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.split_name = split_name
        self.classification_type = classification_type

        self.segments_indices = {s.id: i for i,
                                 s in enumerate(epic_dataset.video_segments)}

        if self.split_name == 'all':
            self.split = np.sort(np.concatenate((self.splits['train'],
                                                 self.splits['validation'],
                                                 self.splits['test'])))
        else:
            self.split = np.sort(self.splits[self.split_name])

        if part != -1:
            start_ind = (part - 1) * 100
            end_ind = np.min((part * 100, len(self.split)))
            self.split = self.split[start_ind:end_ind]

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff]

    def _sample_indices(self, video_segment):
        """
        :param video_segment: VideoSegment
        :return: list
        """

        average_duration = (video_segment.num_frames -
                            self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif video_segment.num_frames > self.num_segments:
            offsets = np.sort(
                randint(video_segment.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, video_segment):
        if video_segment.num_frames > self.num_segments + self.new_length - 1:
            tick = (video_segment.num_frames - self.new_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, video_segment):
        tick = (video_segment.num_frames - self.new_length + 1) / \
            float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])
        return offsets

    def __getitem__(self, index):
        split_index = self.split[index]
        index = self.segments_indices[split_index]
        segment = self.epic_dataset.video_segments[index]

        if self.split_name == 'train':
            segment_indices = self._sample_indices(
                segment) if self.random_shift else self._get_val_indices(segment)
        else:
            segment_indices = self._get_test_indices(segment)

        return self.get(split_index, segment, segment_indices)

    def get_label(self, segment):
        if self.classification_type == 'verb':
            verb_id = segment['verb_class']
            return self.classes_map[verb_id]
        elif self.classification_type == 'noun':
            noun_id = segment['noun_class']
            return self.classes_map[noun_id]
        else:
            verb_id = segment['verb_class']
            noun_id = segment['noun_class']
            return self.classes_map[verb_id][noun_id]

    def get_original_labels(self, index):
        segment = self.epic_dataset.video_segments[index]
        verb_id = segment['verb_class']
        noun_id = segment['noun_class']
        return verb_id, noun_id

    def get(self, index, segment, indices):
        new_indices = np.zeros(0, dtype=int)
        for seg_ind in indices:
            max_frame = np.min((seg_ind + self.new_length, segment.num_frames))
            new_indices = np.hstack(
                (new_indices, np.arange(seg_ind, max_frame, dtype=int)))
        images = self.epic_dataset.load_frames(segment, new_indices)
        process_data = self.transform(images)
        label = self.get_label(segment)
        return index, process_data, label

    def __len__(self):
        return len(self.split)


class EpicTSNTestDataset(data.Dataset):
    def __init__(self,
                 epic_dataset,
                 classes_map,
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 transform=None,
                 random_shift=True,
                 part=-1
                 ):
        self.epic_dataset = epic_dataset
        self.classes_map = classes_map
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.segments_indices = {s.id: i for i,
                                 s in enumerate(epic_dataset.video_segments)}

        self.split = sorted([s.id for s in epic_dataset.video_segments])

        if part != -1:
            start_ind = (part - 1) * 100
            end_ind = np.min((part * 100, len(self.split)))
            self.split = self.split[start_ind:end_ind]

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff]

    def _sample_indices(self, video_segment):
        """
        :param video_segment: VideoSegment
        :return: list
        """

        average_duration = (video_segment.num_frames -
                            self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif video_segment.num_frames > self.num_segments:
            offsets = np.sort(
                randint(video_segment.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, video_segment):
        tick = (video_segment.num_frames - self.new_length + 1) / \
            float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])
        return offsets

    def __getitem__(self, index):
        split_index = self.split[index]
        index = self.segments_indices[split_index]
        segment = self.epic_dataset.video_segments[index]
        segment_indices = self._get_test_indices(segment)
        return self.get(split_index, segment, segment_indices)

    def get(self, index, segment, indices):
        new_indices = np.zeros(0, dtype=int)
        for seg_ind in indices:
            max_frame = np.min((seg_ind + self.new_length, segment.num_frames))
            new_indices = np.hstack(
                (new_indices, np.arange(seg_ind, max_frame, dtype=int)))
        images = self.epic_dataset.load_frames(segment, new_indices)
        process_data = self.transform(images)
        return index, process_data

    def get_original_labels(self, index):
        index = self.segments_indices[index]
        segment = self.epic_dataset.video_segments[index]
        verb_id = segment['verb_class']
        noun_id = segment['noun_class']
        return verb_id, noun_id

    def __len__(self):
        return len(self.split)


class EpicAudioDataset(data.Dataset):
    def __init__(self,
                 audio_dataset_fpath,
                 classes_map,
                 splits,
                 transform=None,
                 split_name='train',
                 classification_type='verb'
                 ):

        self._audio_dataset_fpath = audio_dataset_fpath
        self.audio_dataset = None
        self.segments_indices = {}

        self.classes_map = classes_map
        self.splits = splits
        self.transform = transform
        self.classification_type = classification_type

        if split_name == 'all':
            self.split = np.sort(np.concatenate((self.splits['train'],
                                                 self.splits['validation'],
                                                 self.splits['test'])))
        else:
            self.split = np.sort(self.splits[split_name])

    def __getitem__(self, index):
        if not self.audio_dataset:
            self.audio_dataset = tables.open_file(
                self._audio_dataset_fpath, mode='r')
            self.segments_indices = {s['index']: i for i, s in enumerate(
                self.audio_dataset.root.segments)}

        index = self.split[index]
        index = self.segments_indices[index]
        segment = self.audio_dataset.root.segments[index]
        return self.get(segment)

    def get_images(self, segment):
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        return self.audio_dataset.root.spectrograms[start_idx:end_idx]

    def get_label(self, segment):
        if self.classification_type == 'verb':
            verb_id = segment['verb_class']
            return self.classes_map[verb_id]
        elif self.classification_type == 'noun':
            noun_id = segment['noun_class']
            return self.classes_map[noun_id]
        else:
            verb_id = segment['verb_class']
            noun_id = segment['noun_class']
            return self.classes_map[verb_id][noun_id]

    def get(self, segment):
        label = self.get_label(segment)
        images = self.get_images(segment)

        # TODO: Read not only the first spectrogram (first 4 secs)
        images = images[0, :, :]
        if self.transform:
            images = self.transform(images)
        return segment['index'], images, label

    def __len__(self):
        return len(self.split)


class EpicAudioTestSet(data.Dataset):
    def __init__(self,
                 audio_dataset_fpath,
                 transform=None,
                 ):

        self._audio_dataset_fpath = audio_dataset_fpath
        self.audio_dataset = None
        self.transform = transform

        tmp_testset = tables.open_file(self._audio_dataset_fpath, mode='r')
        self.split = sorted([s['index'] for s in tmp_testset.root.segments])
        self.segments_indices = {s['index']: i for i, s in enumerate(
            tmp_testset.root.segments)}
        self.num_segments = len(self.split)
        try:
            self.labels = {s['index']: (s['verb_class'], s['noun_class'])
                           for s in tmp_testset.root.segments}
        except:
            self.labels = None
        tmp_testset.close()

    def __getitem__(self, index):
        if not self.audio_dataset:
            self.audio_dataset = tables.open_file(
                self._audio_dataset_fpath, mode='r')

        index = self.split[index]
        index = self.segments_indices[index]
        segment = self.audio_dataset.root.segments[index]
        return self.get(segment)

    def get_images(self, segment):
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        return self.audio_dataset.root.spectrograms[start_idx:end_idx]

    def get(self, segment):
        images = self.get_images(segment)

        # TODO: Read not only the first spectrogram (first 4 secs)
        images = images[0, :, :]
        if self.transform:
            images = self.transform(images)
        return segment['index'], images

    def get_original_labels(self, index):
        return self.labels[index]

    def __len__(self):
        return self.num_segments


class EpicEmbeddingsDataset(data.Dataset):

    def __init__(self,
                 embeddings_dataset_fpath,
                 classes_map,
                 splits,
                 transform=None,
                 split_name='train',
                 classification_type='verb',
                 num_segments=3,
                 num_embeddings=25,
                 modalities='rgb+flow+audio_traddil'
                 ):

        self._embeddings_dataset_fpath = embeddings_dataset_fpath
        self.embeddings_dataset = None
        self.segments_indices = {}

        self.classes_map = classes_map
        self.splits = splits
        self.transform = transform
        self.classification_type = classification_type
        self.split_name = split_name
        self.num_segments = num_segments
        self.num_embeddings = num_embeddings
        self.modalities = modalities

        if split_name == 'all':
            self.split = np.sort(np.concatenate((self.splits['train'],
                                                 self.splits['validation'],
                                                 self.splits['test'])))
        else:
            self.split = np.sort(self.splits[self.split_name])

        size = self.num_embeddings // self.num_segments
        self._sampling_sizes = size * np.ones(self.num_segments, dtype=int)
        self._sampling_sizes[-1] += self.num_embeddings % 3
        self._offsets = np.multiply(np.arange(self.num_segments), size)
        self._all_indices = np.arange(self.num_embeddings)

    def get_label(self, segment):
        if self.classification_type == 'verb':
            verb_id = segment.verb_id
            return self.classes_map[verb_id]
        elif self.classification_type == 'noun':
            noun_id = segment.noun_id
            return self.classes_map[noun_id]
        else:
            verb_id = segment.verb_id
            noun_id = segment.noun_id
            return self.classes_map[verb_id][noun_id]

    def __getitem__(self, index):
        if not self.embeddings_dataset:
            self.embeddings_dataset = pickle.load(open(self._embeddings_dataset_fpath, "rb"))

        index = self.split[index]
        segment = self.embeddings_dataset[index]
        return self.get(segment)

    def get(self, segment):

        if self.split_name == 'train':
            indices = self._offsets + [randint(self._sampling_sizes[i]) for i in range(self.num_segments)]
        else:
            indices = self._all_indices

        embeddings = list()
        if 'rgb' in self.modalities:
            embeddings.append(segment.rgb[indices, :].mean(axis=0))
        if 'flow' in self.modalities:
            embeddings.append(segment.flow[indices, :].mean(axis=0))
        if 'audio_vgg' in self.modalities:
            embeddings.append(segment.audio_vgg)
        if 'audio_traddil' in self.modalities:
            embeddings.append(segment.audio_traddil)
        embeddings_vector = self.transform(np.concatenate(embeddings))

        label = self.get_label(segment)
        return segment.segment_id, embeddings_vector, label

    def __len__(self):
        return len(self.split)


class EpicEmbeddingsTestDataset(data.Dataset):

    def __init__(self,
                 embeddings_dataset_fpath,
                 transform=None,
                 classification_type='verb',
                 num_segments=3,
                 num_embeddings=25,
                 modalities='rgb+flow+audio_traddil'
                 ):

        self._embeddings_dataset_fpath = embeddings_dataset_fpath
        self.embeddings_dataset = None

        self.transform = transform
        self.classification_type = classification_type
        self.num_segments = num_segments
        self.num_embeddings = num_embeddings
        self.modalities = modalities
        self._all_indices = np.arange(self.num_embeddings)

        tmp_embeddings_dataset = pickle.load(open(self._embeddings_dataset_fpath, "rb"))
        self.split = sorted([int(segment_id) for segment_id in tmp_embeddings_dataset.keys()])

    def __getitem__(self, index):
        if not self.embeddings_dataset:
            self.embeddings_dataset = pickle.load(open(self._embeddings_dataset_fpath, "rb"))

        index = self.split[index]
        segment = self.embeddings_dataset[index]
        return self.get(segment)

    def get(self, segment):
        embeddings = list()
        if 'rgb' in self.modalities:
            embeddings.append(segment.rgb[self._all_indices, :].mean(axis=0))
        if 'flow' in self.modalities:
            embeddings.append(segment.flow[self._all_indices, :].mean(axis=0))
        if 'audio_vgg' in self.modalities:
            embeddings.append(segment.audio_vgg)
        if 'audio_traddil' in self.modalities:
            embeddings.append(segment.audio_traddil)
        embeddings_vector = self.transform(np.concatenate(embeddings))

        return segment.segment_id, embeddings_vector

    def __len__(self):
        return len(self.split)


class EpicSegment():
    def __init__(self,
                 segment_id,
                 verb_id=None,
                 noun_id=None,
                 rgb=None,
                 flow=None,
                 audio_vgg=None,
                 audio_traddil=None,
                 ):

        self.segment_id = segment_id
        self.verb_id = verb_id
        self.noun_id = noun_id
        self.rgb = rgb
        self.flow = flow
        self.audio_vgg = audio_vgg
        self.audio_traddil = audio_traddil
