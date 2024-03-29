# NFL Big Data Bowl 2024

### Navigation
* local_functions - Contains variety of functions used in each of the other notebooks
* dists - Performs data wrangling and feature engineering
* model - Builds tackle probability model (described below)
* viz - Visualizes tracking data augmented with defender tackle probabilites for example plays

### Approach
The tackle probability model was trained on all defenders for every frame. The target is a binary indicator based on the defender recording a tackle, tackle assist, or forced fumble on the play as denoted in the tackles data. The following features were utilized in the model:

* Defender acceleration
* Defender speed
* Ball carrier acceleration
* Ball carrier speed
* Defender distance to the ball carrier
* Defender orientation relative to the ball carrier
* Defender movement direction relative to the bal carrier
* Defender influence (based on field control model) within a 3 yard radius of the ball carrier
* Indicator of whether the defender was engaged in a block
* Distances from the defender to all other players on the field

A xgboost model was trained using "leave one game out" cross validation. The brier score was 0.08. For defenders that did not make a tackle on a given play, their average tackle probability throughout the play was 9.6%. For defenders that did make a tackle on a given play, their average tackle probability was 31.2%.

### Results
The below play animation shows the tackle probability in action. Offensive players are light blue with the ball carrier highlighted in a darker blue. All defensive players are color scaled from white to dark orange based on their probability of making a tackle throughout the play. All players orientation are display with a blue arrow. All players direction of movement are indicated with an orange arrow sized based on the speed of the player.

You can see several players have probabilities increase dramatically as their path to the ball carrier opens up and they close in on the tackle.

![Animation of Tracking Data with Tackle Probabilities](https://github.com/brandonlester/bdb24/blob/main/figs/animation_2022091101_2501.gif)