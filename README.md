# MachineLearning-AnimeFactory
### Project from the Python Image Processing Cookbook 

```bash
├── SRC
│   ├── discriminator.py ('CNN 2')
│   ├── generator.py ('CNN 1')
├── training-data ('Unpack Dataset Here')
├── AnimeFactory.py ('Run me')
├── LICENSE
├── README.md
└── git-images ('Images for markdown')
```
I love to solve real-world and impacting problems, that's why for this project I will be solving a very important shortage... Anime Characters!

Jokes aside image generation is a super exciting field, solutions are visual so when they work they feel tangible and are great at showcasing the power of machine learning to a non-technical audience. Unfortunately, I often neglect them as they don't often align with the business drivers for my work. They also require considerable amounts of graphical computing, often a bit more than my poor MacBook can push. For this project I will be showing off the power of GAN networks, using as little processing power as I can get away with (Hense the Anime dataset).

Generative adversarial networks are an awesome machine learning concept that pits two CNN models against each other. One model; the generator, counterfeits images while in the other model;the discriminator, tries to detect the fake images. The models drive each other to improve their methods until the fake images are indistinguishable from real images. 

If you want to run the code yourself you will need to download the Anime Face dataset from Kaggle found here: https://www.kaggle.com/splcher/animefacedataset which contains 65,000 scraped faces of various sizes and styles.

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample-training.PNG?raw=true"/>

Unpack this data set in the training-data directory and then run AnimeFactory.py. Depending on your machine this could take up to two hours to finish training.

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/training.PNG"/>

Every Epoch a sample of the faces created will be saved so you can see how the model improves. Here are the results from my training session:

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_1.png?raw=true"/>

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_2.png?raw=true"/>

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_3.png?raw=true"/>

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_4.png?raw=true"/>

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_5.png?raw=true"/>

<img src="https://github.com/MatthewBoyce/MachineLearning-AnimeFactory/blob/main/git-images/sample_15.png?raw=true"/>

