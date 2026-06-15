import os
import sys
import glob
import torch 
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
torchaudio.set_audio_backend("sox_io")

class SignalTrainLA2ADataset(torch.utils.data.Dataset):
    """ SignalTrain LA2A dataset. Source: [10.5281/zenodo.3824876](https://zenodo.org/record/3824876)."""
    def __init__(self, root_dir, subset="train", length=16384, preload=False, half=True, fraction=1.0, use_soundfile=False):
        """
        Args:
            root_dir (str): Path to the root directory of the SignalTrain dataset.
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            use_soundfile (bool, optional): Use the soundfile library to load instead of torchaudio. (Default: False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.use_soundfile = use_soundfile

        # Check if using x_t/y_t format (new format) or original SignalTrain format
        x_t_dir = os.path.join(self.root_dir, "x_t")
        y_t_dir = os.path.join(self.root_dir, "y_t")
        use_xt_yt_format = os.path.isdir(x_t_dir) and os.path.isdir(y_t_dir)
        
        if use_xt_yt_format:
            # New format: x_t and y_t directories
            if self.subset == "full":
                self.target_files = glob.glob(os.path.join(y_t_dir, "*.wav"))
                self.input_files  = glob.glob(os.path.join(x_t_dir, "*.wav"))
            else:
                # For train/val/test, use all files (can be split later if needed)
                self.target_files = glob.glob(os.path.join(y_t_dir, "*.wav"))
                self.input_files  = glob.glob(os.path.join(x_t_dir, "*.wav"))
        else:
            # Original SignalTrain format
            if self.subset == "full":
                self.target_files = glob.glob(os.path.join(self.root_dir, "**", "target_*.wav"))
                self.input_files  = glob.glob(os.path.join(self.root_dir, "**", "input_*.wav"))
            else:
                # get all the target files files in the directory first
                self.target_files = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "target_*.wav"))
                self.input_files  = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "input_*.wav"))

        self.examples = [] 
        self.minutes = 0  # total number of hours of minutes in the subset

        # ensure that the sets are ordered correctly
        self.target_files.sort()
        self.input_files.sort()

        # Helper function for x_t/y_t format
        import re
        def get_seg_id(filename):
            match = re.search(r'(?:seg|input|target)_?(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        # get the parameters 
        if use_xt_yt_format:
            # For x_t/y_t format, extract segment ID and use default params (0, 0.5)
            # Match files by segment ID (e.g., seg0001)
            # Create a mapping of segment ID to file paths
            target_dict = {get_seg_id(f): f for f in self.target_files}
            input_dict = {get_seg_id(f): f for f in self.input_files}
            
            # Match files by segment ID
            matched_pairs = []
            for seg_id in sorted(set(target_dict.keys()) & set(input_dict.keys())):
                matched_pairs.append((target_dict[seg_id], input_dict[seg_id]))
            
            self.target_files = [p[0] for p in matched_pairs]
            self.input_files = [p[1] for p in matched_pairs]
            
            # Use default parameters (limit=0, peak_red=0.5)
            self.params = [(0.0, 0.5) for _ in self.target_files]
        else:
            # Original format: extract from filename
            self.params = [(float(f.split("__")[1].replace(".wav","")), float(f.split("__")[2].replace(".wav",""))) for f in self.target_files]

        # loop over files to count total length
        for idx, (tfile, ifile, params) in enumerate(zip(self.target_files, self.input_files, self.params)):

            # Check file ID matching (different logic for different formats)
            if use_xt_yt_format:
                # For x_t/y_t format, check segment ID
                ifile_id = get_seg_id(ifile)
                tfile_id = get_seg_id(tfile)
                if ifile_id != tfile_id:
                    raise RuntimeError(f"Found non-matching segment ids: {ifile_id} != {tfile_id}! Check dataset.")
            else:
                # Original format: extract from filename like input_123 or target_123
                ifile_id = int(os.path.basename(ifile).split("_")[1])
                tfile_id = int(os.path.basename(tfile).split("_")[1])
                if ifile_id != tfile_id:
                    raise RuntimeError(f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset.")

            md = torchaudio.info(tfile)
            # torchaudio.info returns a tuple (signal_info, encoding_info) for sox backend
            if isinstance(md, tuple):
                num_frames = md[0].length if hasattr(md[0], 'length') else md[0].num_frames
                sample_rate = md[0].rate if hasattr(md[0], 'rate') else md[0].sample_rate
            else:
                num_frames = md.num_frames
                sample_rate = md.sample_rate

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr  = self.load(ifile)
                target, sr = self.load(tfile)

                # Convert to float32 and normalize to [-1, 1] range
                input = input.float() / 32768.0
                target = target.float() / 32768.0

                num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                if input.shape[-1] != target.shape[-1]:
                    print(os.path.basename(ifile), input.shape[-1], os.path.basename(tfile), target.shape[-1])
                    raise RuntimeError("Found potentially corrupt file!")
                if self.half:
                    input = input.half()
                    target = target.half()
            else:
                input = None
                target = None

            # create one entry for each patch
            self.file_examples = []
            num_patches = num_frames // self.length
            # If file is shorter than self.length, use the whole file as one patch
            if num_patches == 0 and num_frames > 0:
                num_patches = 1
                actual_length = num_frames
            else:
                actual_length = self.length
            
            for n in range(num_patches):
                offset = int(n * self.length)
                end = min(offset + actual_length, num_frames) if n == num_patches - 1 and num_patches == 1 else offset + self.length
                self.file_examples.append({"idx": idx, 
                                           "target_file" : tfile,
                                           "input_file" : ifile,
                                           "input_audio" : input[:,offset:end] if input is not None else None,
                                           "target_audio" : target[:,offset:end] if input is not None else None,
                                           "params" : params,
                                           "offset": offset,
                                           "frames" : num_frames})

            # add to overall file examples
            self.examples += self.file_examples
        
        # Check if we have any examples
        if len(self.examples) == 0:
            raise RuntimeError(f"No examples found in dataset. Check that files exist and are long enough (>= {self.length} samples).")
        
        # use only a fraction of the subset data if applicable
        if self.subset == "train":
            classes = set([ex['params'] for ex in self.examples])
            n_classes = len(classes) # number of unique compressor configurations
            
            if n_classes == 0:
                raise RuntimeError("No parameter classes found in dataset. Check that params are being set correctly.")
            
            fraction_examples = int(len(self.examples) * self.fraction)
            n_examples_per_class = int(fraction_examples / n_classes) if n_classes > 0 else 0
            
            # Get sample rate from last processed file
            if len(self.target_files) > 0:
                md = torchaudio.info(self.target_files[0])
                sample_rate = md.sample_rate
            else:
                sample_rate = 192000  # default (192kHz)
            
            n_min_total = ((self.length * n_examples_per_class * n_classes) / sample_rate) / 60 
            n_min_per_class = ((self.length * n_examples_per_class) / sample_rate) / 60 
            print(sorted(classes))
            print(f"Total Examples: {len(self.examples)}     Total classes: {n_classes}")
            print(f"Fraction examples: {fraction_examples}    Examples/class: {n_examples_per_class}")
            print(f"Training with {n_min_per_class:0.2f} min per class    Total of {n_min_total:0.2f} min")

            if n_examples_per_class <= 0: 
                raise ValueError(f"Fraction `{self.fraction}` set too low. No examples selected.")

            sampled_examples = []

            for config_class in classes: # select N examples from each class
                class_examples = [ex for ex in self.examples if ex["params"] == config_class]
                example_indices = np.random.randint(0, high=len(class_examples), size=n_examples_per_class)
                class_examples = [class_examples[idx] for idx in example_indices]
                extra_factor = int(1/self.fraction)
                sampled_examples += class_examples * extra_factor

            self.examples = sampled_examples

        # Get sample rate for minutes calculation
        if len(self.target_files) > 0:
            md = torchaudio.info(self.target_files[0])
            sample_rate = md.sample_rate
        else:
            sample_rate = 192000  # default (192kHz)
        
        self.minutes = ((self.length * len(self.examples)) / sample_rate) / 60 

        # we then want to get the input files
        print(f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            audio_idx = self.examples[idx]["idx"]
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"] 
            input, sr  = torchaudio.load(self.examples[idx]["input_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            target, sr = torchaudio.load(self.examples[idx]["target_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            # Convert to float32 and normalize to [-1, 1] range
            input = input.float() / 32768.0
            target = target.float() / 32768.0
            if self.half:
                input = input.half()
                target = target.half()

        # at random with p=0.5 flip the phase 
        if np.random.rand() > 0.5:
            input *= -1
            target *= -1

        # then get the tuple of parameters
        params = torch.tensor(self.examples[idx]["params"]).unsqueeze(0)
        params[:,1] /= 100

        return input, target, params

    def load(self, filename):
        if self.use_soundfile:
            x, sr = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, sr = torchaudio.load(filename, normalize=False)
        return x, sr


class CustomWaveDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Loudspeaker Distortion Modeling
    Structure:
        root_dir/
            x_t/ (clean audio - input)
            y_t/ (distorted audio - target)
    Files are matched by segment ID (e.g., seg0001 in x_t matches seg0001 in y_t)
    Returns only (input, target) without conditioning parameters.

    Train/Test Split:
    - Files are split based on segment ID to ensure consistent train/test separation
    - Default split: 80% train, 20% test (can be adjusted via train_test_split_ratio)
    - Split is deterministic (sorted by seg_id) to ensure reproducibility

    Data Augmentation (train subset only):
    - Volume Scaling: 진폭을 [volume_scale_min, volume_scale_max] 범위에서 무작위 스케일링.
      input / target 모두에 동일한 스케일 팝터를 적용하여
      비선형 왼곡률이 입력 신호 진폭에 따라 변하는 특성을 학습할 수 있도록 함.
    """
    def __init__(
        self,
        root_dir,
        subset                = "train",
        length                = 16384,
        preload               = False,
        half                  = True,
        fraction              = 1.0,
        train_test_split_ratio= 0.8,
        target_sample_rate    = 192000,
        # ---- 증강(Augmentation) 옵션 ----
        augment               = True,    # True이면 train subset에서만 증강 적용
        volume_scale_min      = 0.25,    # 스케일 팝터 최소값 (1.0 = 원본 진폭)
        volume_scale_max      = 1.0,     # 스케일 팝터 최대값
    ):
        self.root_dir               = root_dir
        self.subset                 = subset  # "train", "val", "test", or "full"
        self.length                 = length
        self.preload                = preload
        self.half                   = half
        self.fraction               = fraction
        self.train_test_split_ratio = train_test_split_ratio
        self.target_sample_rate     = target_sample_rate
        self.resampler              = None  # Will be initialized when needed
        # ---- 증강(Augmentation) 설정 ----
        self.augment           = augment
        self.volume_scale_min  = volume_scale_min
        self.volume_scale_max  = volume_scale_max

        # Define paths (x_t/y_t format)
        self.input_dir = os.path.join(self.root_dir, "x_t")
        self.target_dir = os.path.join(self.root_dir, "y_t")

        # Get all files
        all_input_files = glob.glob(os.path.join(self.input_dir, "*.wav"))
        all_target_files = glob.glob(os.path.join(self.target_dir, "*.wav"))

        # Helper function to extract segment ID
        import re
        def get_seg_id(filename):
            match = re.search(r'(?:seg|input|target)_?(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else -1

        # Create dictionaries mapping seg_id to file path
        input_dict = {get_seg_id(f): f for f in all_input_files if get_seg_id(f) >= 0}
        target_dict = {get_seg_id(f): f for f in all_target_files if get_seg_id(f) >= 0}

        # Match files by segment ID and sort by seg_id for deterministic split
        matched_pairs = []
        for seg_id in sorted(set(input_dict.keys()) & set(target_dict.keys())):
            matched_pairs.append((seg_id, input_dict[seg_id], target_dict[seg_id]))

        # Sort by segment ID for consistent train/test split
        matched_pairs.sort(key=lambda x: x[0])

        # Split into train/test based on subset parameter
        if self.subset == "full":
            # Use all files
            filtered_pairs = matched_pairs
        elif self.subset in ["train", "val"]:
            # Use train portion (80% by default)
            split_idx = int(len(matched_pairs) * self.train_test_split_ratio)
            filtered_pairs = matched_pairs[:split_idx]
        elif self.subset == "test":
            # Use test portion (20% by default)
            split_idx = int(len(matched_pairs) * self.train_test_split_ratio)
            filtered_pairs = matched_pairs[split_idx:]
        else:
            raise ValueError(f"Unknown subset: {self.subset}. Must be 'train', 'val', 'test', or 'full'")

        self.input_files = [p[1] for p in filtered_pairs]
        self.target_files = [p[2] for p in filtered_pairs]

        if len(self.input_files) == 0:
            raise RuntimeError(f"No matching file pairs found in {self.input_dir} and {self.target_dir} for subset '{self.subset}'")
        
        # Print split information
        total_files = len(matched_pairs)
        train_files = int(total_files * self.train_test_split_ratio)
        test_files = total_files - train_files
        print(f"Dataset split: {total_files} total files -> {train_files} train, {test_files} test")
        print(f"Using subset '{self.subset}': {len(self.input_files)} files")

        self.examples = []
        self.minutes = 0

        # Process files
        for idx, (ifile, tfile) in enumerate(zip(self.input_files, self.target_files)):
            # Verify segment ID match
            ifile_id = get_seg_id(ifile)
            tfile_id = get_seg_id(tfile)
            if ifile_id != tfile_id:
                raise RuntimeError(f"Found non-matching segment ids: {ifile_id} != {tfile_id}!")

            # Get file info
            md = torchaudio.info(tfile)
            if isinstance(md, tuple):
                num_frames = md[0].length if hasattr(md[0], 'length') else md[0].num_frames
                sample_rate = md[0].rate if hasattr(md[0], 'rate') else md[0].sample_rate
            else:
                num_frames = md.num_frames
                sample_rate = md.sample_rate

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.input_files):3d} ...\r")
                sys.stdout.flush()
                input, sr = torchaudio.load(ifile, normalize=False)
                target, sr = torchaudio.load(tfile, normalize=False)

                # Resample if needed
                if sr != self.target_sample_rate:
                    resampler = T.Resample(sr, self.target_sample_rate)
                    input = resampler(input)
                    target = resampler(target)
                    sr = self.target_sample_rate

                # Convert to float32 and normalize to [-1, 1] range
                input = input.float() / 32768.0
                target = target.float() / 32768.0

                if self.half:
                    input = input.half()
                    target = target.half()
            else:
                input = None
                target = None

            # Create patches
            self.file_examples = []
            if num_frames < self.length:
                # If file is shorter than length, use it as a single patch
                self.file_examples.append({
                    "idx": idx,
                    "target_file": tfile,
                    "input_file": ifile,
                    "input_audio": input,
                    "target_audio": target,
                    "offset": 0,
                    "frames": num_frames
                })
            else:
                # Split into multiple patches
                for n in range(num_frames // self.length):
                    offset = int(n * self.length)
                    self.file_examples.append({
                        "idx": idx,
                        "target_file": tfile,
                        "input_file": ifile,
                        "input_audio": input[:, offset:offset+self.length] if input is not None else None,
                        "target_audio": target[:, offset:offset+self.length] if target is not None else None,
                        "offset": offset,
                        "frames": num_frames
                    })

            self.examples += self.file_examples

        if len(self.examples) == 0:
            raise RuntimeError(f"No examples found in dataset. Check that files exist and are long enough (>= {self.length} samples).")

        # Apply fraction if needed (for training subset)
        if self.subset == "train" and self.fraction < 1.0:
            n_examples = int(len(self.examples) * self.fraction)
            if n_examples > 0:
                indices = np.random.choice(len(self.examples), n_examples, replace=False)
                self.examples = [self.examples[i] for i in indices]

        # Calculate total minutes
        if len(self.target_files) > 0:
            md = torchaudio.info(self.target_files[0])
            if isinstance(md, tuple):
                original_sr = md[0].sample_rate
            else:
                original_sr = md.sample_rate
            # Use target_sample_rate for calculation (after resampling)
            sample_rate = self.target_sample_rate
        else:
            sample_rate = self.target_sample_rate

        self.minutes = ((self.length * len(self.examples)) / sample_rate) / 60
        print(f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"]
            input, sr = torchaudio.load(
                self.examples[idx]["input_file"],
                num_frames=self.length,
                frame_offset=offset,
                normalize=False
            )
            target, sr = torchaudio.load(
                self.examples[idx]["target_file"],
                num_frames=self.length,
                frame_offset=offset,
                normalize=False
            )
            
            # Resample if needed
            if sr != self.target_sample_rate:
                if self.resampler is None or self.resampler.orig_freq != sr:
                    self.resampler = T.Resample(sr, self.target_sample_rate)
                input = self.resampler(input)
                target = self.resampler(target)
                sr = self.target_sample_rate
            
            # Convert to float32 and normalize to [-1, 1] range
            input = input.float() / 32768.0
            target = target.float() / 32768.0
            if self.half:
                input = input.half()
                target = target.half()

        # Volume Scaling 증강 로직 (augment=True 일 때만 동작)
        if self.augment:
            scale = torch.empty(1).uniform_(self.volume_scale_min, self.volume_scale_max).item()
            # 동일한 스케일 팩터를 input과 target에 곱하여 비선형 왜곡률 학습
            input  = input  * scale
            target = target * scale

        # IMPORTANT: Return only input and target (NO params)
        return input, target