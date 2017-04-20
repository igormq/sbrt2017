from datasets import DatasetParser

import os
import re
import librosa
import codecs

regex = r"User\s+Name\:[\s]*(?P<speaker>.*)[\n]+.*[\n]+Gender\:[\s]*(?P<gender>[a-zA-Z]+)[\w\r\s\n:\/]+Pronunciation dialect\:\s+(?P<dialect>.*)"


class VoxForge(DatasetParser):
    """ VoxForge (only portuguese brazilian audio files) dataset reader and parser

    More about the dataset: http://www.voxforge.org/
    """

    IGNORED_LIST = ['Marcelo-20131106-iqc',
                    'anonymous-20140619-wcy',
                    'ThiagoCastro-20131129-qpn',
                    'anonymous-20131016-uzv']

    def __init__(self, dataset_dir=None, name='voxforge', **kwargs):

        dataset_dir = dataset_dir or 'data/voxforge'

        super(VoxForge, self).__init__(dataset_dir, name, **kwargs)

        if (self.dataset_dir is not None and
            os.path.isdir(os.path.join(self.dataset_dir, 'files'))):

            self.dataset_dir = os.path.join(self.dataset_dir, 'files')

    def _iter(self):
        for speaker_path in os.listdir(self.dataset_dir):

            if speaker_path in self.IGNORED_LIST:
                continue

            root_path = os.path.join(
                os.path.abspath(self.dataset_dir), speaker_path)

            if not os.path.isdir(os.path.join(root_path)):
                continue

            labels_file = os.path.join(root_path, 'etc', 'PROMPTS')

            if not os.path.exists(labels_file):
                labels_file = os.path.join(root_path, 'PROMPTS')

            speaker_info_file = os.path.join(root_path, 'etc', 'README')

            if not os.path.exists(speaker_info_file):
                speaker_info_file = os.path.join(root_path, 'README')

            with open(speaker_info_file) as f:
                info_text = f.read()

            pattern = re.compile(regex, re.MULTILINE | re.UNICODE)

            info = list(re.finditer(pattern, info_text))[0].groupdict()

            gender = info['gender'][0].lower()
            speaker_id = info['speaker']

            for line in codecs.open(labels_file, 'r', encoding='utf8'):
                split = line.strip().split()
                file_id = split[0].split('/')[-1]

                label = ' '.join(split[1:]).lower()

                audio_file = os.path.join(root_path, 'wav', file_id) + '.wav'

                if not os.path.exists(audio_file):
                    audio_file = os.path.join(root_path, file_id) + '.wav'

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'gender': gender,
                       'speaker': speaker_id}

    def _report(self, dl):
        args = len(dl['audio']), sum(dl['duration']),
        len(set(dl['speaker'])),
        100 * (sum([1 for g in dl['gender'] if g == 'f']) /
               (1.0 * len(dl['gender']))),
        100 * (sum([1 for s in dl['speaker'] if s == 'anonymous']) /
               (1.0 * len(dl['speaker'])))

        report = '''General information
                Number of utterances: %d
                Total size (in seconds) of utterances: %.f
                Number of speakers: %d
                %% of female speaker: %.2f%%
                Anonymous speaker: %.2f%%''' % (args)

        return report
