from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

DURATIONS = [
    "short",
    "medium",
    "long",
]

DOMAINS = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

def get_dimension_rating(data_path):
    data = load(data_path)

    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ["overall"]:
        duration_rating[duration] = {
            "overall": "",
            "domain": {k: [] for k in DOMAINS},
            "sub_category": {k: [] for k in SUB_CATEGORIES},
            "task_type": {k: [] for k in TASK_CATEGORIES}
        }

    for i in range(len(data)):

        domain = data.iloc[i]['domain']
        sub_category = data.iloc[i]['sub_category']
        task_category = data.iloc[i]['task_type']

        duration = data.iloc[i]['duration']
        duration_rating[duration]["domain"][domain].append(data.iloc[i]['score'])
        duration_rating[duration]["sub_category"][sub_category].append(data.iloc[i]['score'])
        duration_rating[duration]["task_type"][task_category].append(data.iloc[i]['score'])

        duration_rating["overall"]["domain"][domain].append(data.iloc[i]['score'])
        duration_rating["overall"]["sub_category"][sub_category].append(data.iloc[i]['score'])
        duration_rating["overall"]["task_type"][task_category].append(data.iloc[i]['score'])
    
    for duration in DURATIONS + ["overall"]:

        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

    return duration_rating


def extract_option(model, input_item, dataset_name):
    options = input_item['question'].split('\n')[1:]
    for id, option in enumerate(options):
        option_id = chr(ord('A') + id) + '.'
        if option.find(option_id) >= 0:
            input_item[chr(ord('A') + id)] = option[option.find(option_id) + len(option_id):].strip('. \n')
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def prepare_dataset(self, dataset_name='Video-MME', repo_id='modelscope/Video-MME'):
    def check_integrity(pth):
        data_file = osp.join(pth, f'{dataset_name}.tsv')

        if not os.path.exists(data_file):
            return False
        
        if md5(data_file) != self.MD5:
            return False
        data = load(data_file)
        for video_pth in data['video_path']:
            if not osp.exists(osp.join(pth, video_pth)):
                return False
        return True

    dataset_path = os.path.expanduser(f"~/LMUData/{dataset_name}")
    
    if not check_integrity(dataset_path):
        def unzip_hf_zip(pth):
            import zipfile
            base_dir = pth
            target_dir = os.path.join(pth, 'video/')
            zip_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith('.zip') and file.startswith('video')]
            zip_files.sort()
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                for zip_file in zip_files:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(target_dir, os.path.basename(member)), "wb")
                                with source, target:
                                    target.write(source.read())
                print('The video file has been restored and stored from the zip file.')
            else:
                print('The video file already exists.')
            
            subtitle_zip_file = os.path.join(base_dir, 'subtitle.zip')
            subtitle_target_dir = os.path.join(base_dir, 'subtitle')

            if not os.path.exists(subtitle_target_dir):
                os.makedirs(subtitle_target_dir, exist_ok=True)
                with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # Check if the member is a file (not a directory)
                        if not member.endswith('/'):
                            # Extract the file to the specified directory
                            source = zip_ref.open(member)
                            target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), "wb")
                            with source, target:
                                target.write(source.read())
                print('The subtitle file has been restored and stored from the zip file.')
            else:
                print('The subtitle file already exists.')
            
        def generate_tsv(pth):

            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if os.path.exists(data_file) and md5(data_file) == self.MD5:
                return
            
            data_file = pd.read_parquet(os.path.join(pth, 'videomme/test-00000-of-00001.parquet'))
            data_file = data_file.assign(index=range(len(data_file)))
            data_file['video'] = data_file['videoID']
            data_file['video_path'] = data_file['videoID'].apply(lambda x: f'./video/{x}.mp4')
            data_file['subtitle_path'] = data_file['videoID'].apply(lambda x: f'./subtitle/{x}.srt')
            data_file['candidates'] = data_file['options'].apply(lambda x: x.tolist())

            data_file = data_file[['index', 'video', 'video_path', 'duration', 'domain', 'candidates',
                                    'sub_category', 'task_type', 'subtitle_path', 'question', 'answer']]

            data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

        subprocess.run(['modelscope', 'download', '--dataset', repo_id, '--local_dir', dataset_path])
        unzip_hf_zip(dataset_path)
        generate_tsv(dataset_path)

    data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

    return dict(data_file=data_file, root=dataset_path)