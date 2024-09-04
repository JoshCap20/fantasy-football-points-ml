"""
Config Module

Contains general configuration settings for the project.
"""

"""
Year ranges for training and testing data.

The training data is used to train the models, while the testing data is used to evaluate the models.
The data will be automatically scraped if not already cached for that year.
"""
TRAIN_YEARS: list[int] = [2022]
TEST_YEARS: list[int] = [2023]


"""
Setting this to true will log more information, in addition to outputting the debug logs to debug.log.
"""
DEBUG: bool = True


"""
Don't modify this unless you know what you're doing.

Adding K or P won't work since we don't have kicking stats scraped,
so you'll be estimating their performance based on other position stats (pointless).
"""

POSITIONS: list[str] = ["QB", "RB", "TE", "WR"]
# TODO: Add support for P/K (need kicking/punting stats)
