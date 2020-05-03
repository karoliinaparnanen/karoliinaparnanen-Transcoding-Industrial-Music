<img src="Transcoding%20Industrial%20Music/coverphoto.jpg" width="1920">

# Transcoding Industrial Music

### Table of Contents
1. [Artefact](#Artefact)
    1. [Metadata](#Metadata)
2. [Research](#Research)
3. [Transcoding](#Transcoding)
    1. [Transcoding No. 1](#Transcoding-no-1)
    2. [Transcoding Tool](#Transcoding-tool)
4. [Reflection](#Reflection)
5. [Outcome](#Outcome)
6. [Conclusion](#Conclusion)
7. [Bibliography](#Bibliography) 

# Artefact
The artefacts in question are two industrial music complilations called "Extreme Art 1: Industrial Compilation" published in 1987 and "Extreme Art 2: Industrial Compilation" published in 1988 by Bizarr Verlag in Germany. The VHS tapes include music videos, live music performances and art performances from various artists from around the world, focusing on European artists. 

# Metadata

## Artefact 1 – Extreme Art 1: Industrial Music Compilation

| Tag | Data |  
|--|--|
**IISG Call Number** | [CSD BG V2/410 ](https://search.iisg.amsterdam/Record/1046576)
| **Physical Description** | VHS. 60 min. 
| **Type** | Visual Document
| **Medium** | Video
| **Materials** | Plastic, magnetic tape, paper.
| **Date** | 1987
|![VHS cover](Transcoding%20Industrial%20Music/extremeart6.jpg)| The washed out pink, purple, blue, white and black shaded cover image shows a video still closeup of a man's face. From the camera's perspective, the man is looking down on the viewer with half-closed eyelids. His mouth is relaxed. The strong contrast, half washed out face and pixels that remind one of an 80's television screen make the image of him appear dreamlike, ecstatic and intentionally grunfy and careless. The title "Extreme Art Industrial Compilation" is spread through the top part with a black type, whose serifs are sharp like knives. The number "1" stands isolated in a bigger size on the lower left corner. The dimensions of the VHS tape inside are 18.7 × 10.2 × 2.5 cm. 

## Artefact 2 – Extreme Art 2: Industrial Music Compilation

| Tag | Data |  
|--|--|
**IISG Call Number** | [CSD BG V2/411](https://search.iisg.amsterdam/Record/1046580)
| **Physical Description** | VHS. 60 min. 
| **Type** | Visual Document
| **Medium** | Video
| **Materials** | Plastic, magnetic tape, paper.
| **Date** | 1988
|![VHS cover](Transcoding%20Industrial%20Music/extremeart.jpg)| The washed out purple, blue, white and black shaded cover image shows a a video still of a photographic, ghost-like figure approaching the viewer with open hands. The figure is heavily lit from the back, so any details of the figure are not visible. The black figure's silhouette is visible over a light, almost white background. the image quality resembles that of an 80's television screen capture. The title "Extreme Art Industrial Compilation" is spread through the top part with a black type, whose serifs are sharp like knives. The number "2" stands isolated in a bigger size on the lower left corner. The dimensions of the VHS tape inside are 18.7 × 10.2 × 2.5 cm. 

# Research

## Ideology of Industrial Music
The birth of industrial music was a response to an age in which the access and control of information were becoming the primary tools of power. As an artistic movement, it has formed due to the simultaneous fear and fascination of **how the information revolution and the effects of the age of mechanical reproduction affects the human condition and social consciousness.** As the purpose of all art is to expand the beholder's perception, so was the purpose of industrial music to help the listener achieve a better understanding and awareness of **how the advent of technology influences his view of the world.** **Industrial musicians “create particular modernist aesthetics that attempt **to comprehend and comment on what came to be known as the ‘modern crisis’ of the twentieth century.** Instead of rallying youth behind political slogans, **industrial artists preferred to “decondition” the individual listener by confronting taboos.** 

## Visual aesthetics of Industrial Music
Industrial music draws on harsh, transgressive or provocative sounds and themes. It is a form of **anti-music**, descendant of **da-da**, **surrealist** and **performance art** and an artistic yet intellectual manifestation of **nihilism**.  The industrial musicians created a particular modernist aesthetics that attemted to comprehend and comment on what came to be known as the "modern" crisis of the twentieth century". 

## Auditory Aesthetics of Industrial Music
At its birth, the genre of industrial music was different from any other music, and its **use of technology and disturbing lyrics and themes to tear apart preconceptions about the necessary rules of musical** form supports seeing industrial music as modernist music.  Industrial music also got known for its unconventional use of instruments, often including self-modified instruments and so-called **non-intruments** or found objects, which could be any objects the artist felt fitting to create auditory atmospheres. Use of drones and noise was popular. It often combined electronic instruments, such as **the keyoard and computer programming with electric guitars, bass, drums and vocals**. Lyrics were often present, but not easily understood word by word. The literal text of the lyrics was less important than creating an atmosphere were the listener could let go of their preconceptions of music and taste. 

Industrial music's anarchist atmosphere enabled a rich variety of subgenres to develop, such as substyles inspirted by industrial music include dark ambient, power electronics, Japanoise, neofolk, electro-industrial, electronic body music, industrial hip hop, industrial rock, industrial metal, industrial pop, martial industrial, power noise, and witch house.

## About the Publisher
Bizarr Verlag was a Munich based label owned by **Markus Schmölz**, which specialized in audio tour recordings in big cities around the world. Their sublabel AudioTours Streetlife published a series of field recordings. Nowadays the label is an online store selling cards, stickers and other merchandise. 
[bizarrverlag.de](https://www.bizarrverlag.com/). 

# Transcoding
In my transcoding I wanted to research how a time-specific subculture could potentially be preserved with contemporary mediums. When the spirit of an aesthetical and political movement is taken out of its context, how will this affect the way it is perceived? How will the viewer exprience the industrial music scene through the context of a video game? On my transcoding I focused on two methods, preservation and transformation. On the other hand I wish to preserve the spirit of industrial music in a medium that is more approachable for the current generations, but on the other hand I was interested in exploring how the medium transforms and mutates the original in the process of translation. How does the death of a medium, such as the VHS tape, affect the relevance of historical movements, that were mostly recorded in mediums that are not popular anymore, and whose content can not be found through modern mediums, such as the internet? Could giving an internet life to an analog medium enhance its relevane in the current age?

## Transcoding Tool
On the transcoding tool I focused on the part of transformation. What would happen to the original sounds if they were fed into an artificial intelligence algorithm that would try to recreate the samples based on the industrial music material. 

### Sample VAE - A multi-pupose tool for sound design and music production
For my transcoding tool I found a deep learning-based tool made originally by Max Frenzel. The original documentation can be found on his [GitHub page](https://github.com/maxfrenzel/SampleVAE). The tool allows for various types of new sample generation, as well as sound classification, and searching for similar samples in an existing sample library. The coding language used is Python, and the scripting is executed via Terminal. The deep learning part is implemented in TensorFlow and consists mainly of a Variational Autoencoder (VAE) with Inverse Autoregressive Flows (IAF) and an optional classifier network on top of the VAE's encoder's hidden state. You can read further [about the SampleVAE method in this article.](https://towardsdatascience.com/samplevae-a-multi-purpose-ai-tool-for-music-producers-and-sound-designers-e966c7562f22?source=friends_link&sk=588d13c6080568aca63f98e4d3835c87)

The following instructions follow the original guidance of Max Frenzel, with some additions that I found helpful myself when using the code.

### Making a dataset for training
Use the `make_dataset.py` script to generate a new dataset for trainig. The main parameters are `data_dir` and `dataset_name`. The former is a directory (or multiple directories) in which to look for samples (files ending in .wav, .aiff, or .mp3; only need to specify root directory, the script looks into all sub directories). The latter should a unique name for this dataset.
For example, to search for files in the directories `/Users/Shared/Decoded Forms Library/Samples` and `/Users/Shared/Maschine 2 Library`, and create the dataset called `NativeInstruments`, run the command

```
 python make_dataset.py --data_dir '/Users/Shared/Decoded Forms Library/Samples' '/Users/Shared/Maschine 2 Library' --dataset_name NativeInstruments
```

By default, the data is split randomly into 90% train and 10% validation data. The optional `train_ratio` parameter (defaults to 0.9) can be used to specify a different split.

#### Creating a dataset that includes class information
To add a classifier to the model, use `make_dataset_classifier.py` instead. This script works essentially in the same way as `make_dataset.py`, but it treats the immediate sub-directories in `data_dir` as class names, and assumes all samples within them belong to that resepective class.

Currently only simple multiclass classification is supported. There is also no weighting of the classes happening so you should make sure that classes are as balanced as possible. Also, the current version does the train/validation split randomly; making sure the split happens evenly across classes is a simple future improvement.

### Training a model
To train a model, use the `train.py` script. The main parameters of interest are `logdir` and `dataset`.

`logdir` specifies a unique name for the model to be trained, and creates a directory in which model checkpoints and other files are saved. Training can later be resumed from this.

`dataset` refers to a dataset created through the `make_dataset.py` script, e.g. `IndustrialDataset` in the example above.

To train a model called `model_Industrial` on the above dataset, use

```
  python train.py --logdir model_Industrial --dataset IndustrialDataset
```

On first training on a new dataset, all the features have to be calculated. This may take a while. When restarting the training later, or training a new model with same dataset and audio parameters, existing features are loaded.

Training automatically stops when the model converges. If no improvement is found on the validation data for several test steps, the learning rate is lowered. Once it goes below a threshold, training stops completely.
Alternatively one can manually stop the training at any point.

When resuming a previously aborted model training, the dataset does not have to be specified, the script will automatically use the same dataset (and other audio and model parameters).

If the dataset contains classification data, a confusion matrix is plotted and stored in `logdir` at every test step.

### Pre-trained Models
Three trained models are provided:

`model_general`: A model trained on slightly over 60k samples of all types. This is the same dataset that was used in my NeuralFunk project (https://towardsdatascience.com/neuralfunk-combining-deep-learning-with-sound-design-91935759d628). This model does not have a classifier.

`model_drum_classes`: A model trained on roughly 10k drum sounds, with a classifier of 9 different drum classes (e.g. kick, snare, etc).

`model_drum_machines`: A model trained on roughly 4k drum sounds, with a classifier of 71 different drum machine classes (e.g. Ace Tone Rhythm Ace, Akai XE8, Akai XR10 etc). Note that this is a tiny dataset with a very large number of classes, each only containing a handful of examples. This model is only included as an example of what's possible, not as a really useful model in itself.

### Running the sound sample tool with a trained model
To use the sample tool, start a python environment and run

```
from tool_class import *
```

You can now instantiate a SoundSampleTool class. For example to instatiate a tool based on the above model, run.

```
tool = SoundSampleTool('model_Industrial', library_dir='/Users/MySampleLibrary')
```

The parameter `library_dir` is optional and specifies a sample library root directory, `/Users/MySampleLibrary` in the example above. It is required to perform similarity search on this sample library. If specified, an attempt is made to load embeddings for this library. If none are found, new embeddings are calculated which may take a while (depending on sample library size).

Once completely initialised, the tool can be used for sample generation and similarity search.

#### Generating samples
To generate new samples, use the `generate` function.

To sample a random point in latent space, decode it, and store the audio to `generated.wav`, run

```
tool.generate(out_file='generated.wav')
```

To encode one or multiple files, pass the filenames as a list of strings to the `audio_files` parameter. If the parameter `weights` is not specified, the resulting embeddings will be averaged over before decoding into a single audio file. Alternatively, a list of numbers can be passed to `weights` to set the respective weights of each input sample in the average. By default, the weights get normalised. E.g. the following code combines an example kick and snare, with a 1:2 ratio:

```
tool.generate(out_file='generated.wav', audio_files=['/Users/Shared/Maschine 2 Library/Samples/Drums/Kick/Kick Accent 1.wav','/Users/Shared/Decoded Forms Library/Samples/Drums/Snare/Snare Anodyne 3.wav'], weigths=[1,2])
```

Weight normalisation can be turned off by passing `normalize_weights=False`. This allows for arbitrary vector arithmetic with the embedding vectors, e.g. using a negative weight to subtract one vector from another.

Additionally the `variance` parameter (default: 0) can be used to add some Gaussian noise before decoding, to add random variation to the samples.

#### Note on sample length and audio segmentation
Currently, the tool/models treat all samples as 2 second long clips. Shorter files get padded, longer files crop.

For the purpose of building the library, an additional parameter, `library_segmentation`, can be set to `True` when initialising the tool. If `False`, files in the library are simply considered as their first 2 second. However, if `True`, the segments within longer files are considered as individual samples for the purpose of the library and similarity search.
Note that while this is implemented and technically working, the segmentation currently seems too sensitive.

tool.generate(out_file='generated.wav')
To encode one or multiple files, pass the filenames as a list of strings to the audio_files parameter. If the parameter weights is not specified, the resulting embeddings will be averaged over before decoding into a single audio file. Alternatively, a list of numbers can be passed to weights to set the respective weights of each input sample in the average. By default, the weights get normalised. E.g. the following code combines an example kick and snare, with a 1:2 ratio:

tool.generate(out_file='generated.wav', audio_files=['/Users/Shared/Maschine 2 Library/Samples/Drums/Kick/Kick Accent 1.wav','/Users/Shared/Decoded Forms Library/Samples/Drums/Snare/Snare Anodyne 3.wav'], weigths=[1,2])
Weight normalisation can be turned off by passing normalize_weights=False. This allows for arbitrary vector arithmetic with the embedding vectors, e.g. using a negative weight to subtract one vector from another.

Additionally the variance parameter (default: 0) can be used to add some Gaussian noise before decoding, to add random variation to the samples.

### Note on sample length and audio segmentation
Currently, the tool/models treat all samples as 2 second long clips. Shorter files get padded, longer files crop.

For the purpose of building the library, an additional parameter, library_segmentation, can be set to True when initialising the tool. If False, files in the library are simply considered as their first 2 second. However, if True, the segments within longer files are considered as individual samples for the purpose of the library and similarity search. Note that while this is implemented and technically working, the segmentation currently seems too sensitive.

<summary>CLICK ME</summary>

