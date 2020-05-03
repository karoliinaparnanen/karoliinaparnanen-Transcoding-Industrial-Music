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

### Description
You might have made multiple coding tools during the semester. Or the transcoding tool might be part of a specific transcoding experiment itself. It's up to you to define the linear flow of the Readme. Just like the transcoding of media itself, the coding tool should be well documented. So if you are using a coding tool in your first experiment, include the documentation of the coding tool **before** you include the results, etc. 
