{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "#Predicting College Admission with Binary Classifiers\n",
        "\n",
        "By: Michelle Badalov, Tamar Kellner, Zeynep Yilmazcoban, and Saima Ahmed"
      ],
      "metadata": {
        "id": "EIikzSw8Nj3D"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%%shell\n",
        "jupyter nbconvert --to html /content/422_Final_Project.ipynb"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ak3IY--_fZwm",
        "outputId": "b4294e5c-cc60-4a7b-ba5d-3f0bc4a54d6b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[NbConvertApp] Converting notebook /content/422_Final_Project.ipynb to html\n",
            "[NbConvertApp] Writing 455954 bytes to /content/422_Final_Project.html\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 101
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Introduction\n",
        "\n",
        "For our final project, we will attempt to predict chances of university admissions. The data we will be using is the \"Data for Admission in the University\" dataset from Kaggle. You can check it out from this link: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university."
      ],
      "metadata": {
        "id": "js8mP4nTHxIs"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Motivation\n",
        "\n",
        "Our main motivation is the financial burden of college applications on students. College Board recommends students to apply to 5 to 8 colleges. On average, the cost of one application is around 50 \\$ and can go as high as 80 \\$. This can add up to a total cost of 250\\$ - 640\\$ which is a lot to handle for a high school student. As we all know, college admissions can be tricky and stressful. Although students can roughly have an idea about their chances of being admitted to a particular university, decisions are often unexpected. Having a more accurate way to predict chances of admissions can help students make better decisions while deciding which schools to apply to, thus maximize their chances with the cost they pay.\n",
        "\n",
        "To learn more about how the College Board suggests students should narrow down the list of schools they will apply to, visit https://counselors.collegeboard.org/college-application/how-many.\n"
      ],
      "metadata": {
        "id": "ru8pDnsGH55b"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Binary Classification"
      ],
      "metadata": {
        "id": "jRpkhx4QH-Kt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Binary classification is one of the most popular implementations of machine learning. It is a supervised learning algorithm which classifies a set of examples into two groups or classes. The prediction is based on a the chosen binary classification algorithm. We will explore several of these algorithms such as:\n",
        "1. Logistic Regression\n",
        "2. K-Nearest Neighbors\n",
        "3. Decision Tree Classifier\n",
        "\n",
        "Our problem can easily be cast as a binary classification task.\n",
        "\n",
        "Our dataset includes a 'Chance of Admit' column which represents the probability that a student will be granted admission to the university.\n",
        "We can map these percentages to another column which marks a row as 0 if the percentage is less than 0.5, and 1 otherwise, where 0 represents that the student is unlikely to be admitted and 1 is the contrary.\n",
        "\n",
        "Binary Classification has a broad scope of applications beyond admissions statistics. This includes, but is not limited to email spam classification, credit card fraud, quality control, and more.\n",
        "\n",
        "To learn more about Binary Classification check out:\n",
        "\n",
        "https://www.sciencedirect.com/topics/computer-science/binary-classification\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "cZx3LFOF9F3_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Preparing the dataset\n",
        "\n",
        "In this step, we will adjust and prepare our dataset so that we can apply our algorithms and produce the most accurate results.\n",
        "To assist us with preprocessing, we load the following packages:"
      ],
      "metadata": {
        "id": "diImNmEGU3OQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "ZHpZCzoO138v",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "70aba79f-bc8e-48b0-9e64-2724217ea50c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "We then import our data using pandas' read_csv function.\n",
        "To make our lives easier, we first trim any following white space from the column names. Then, we create our truth column, y, which filters the \"Chance of Admit\" column so that it represents whether students have >= 50% chance of admission. The dataset includes a \"Serial No.\" column which we will be dropping since it is irrelevant. Later, we also drop the 'Chance of Admit' column from the input data as it represents the true class."
      ],
      "metadata": {
        "id": "tlVmL_ctIHeK"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aGigODCH7WXm"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv('./drive/MyDrive/422 Final Project/adm_data.csv')\n",
        "\n",
        "# Deletes space after column names\n",
        "df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ':'LOR'}, inplace = True)\n",
        "\n",
        "# Adds binary column which represents whether student has >=50% of admission.\n",
        "y = np.where(df['Chance of Admit'] > 0.5, 1, 0)\n",
        "\n",
        "# Drops irrelevant serial number attribute.\n",
        "X = df.drop(columns=['Serial No.'])\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now that our dataset is cleaned and organized, we can split it into test and training. We use the train_test_split function from sklearn to do this.\n",
        "\n",
        "This function will give us our full X training data, full X test data, y training data, and y test data."
      ],
      "metadata": {
        "id": "8gNZmYopduVY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "2S7nxyIIdu-Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "To perform some brief optimization on the hold out threshold, we tested the performance of logistic regression on various hold out ratios. As seen in the plot of accuracies below, a threshold of 0.4 yields the best performance. Therefore, we will set our hold out ratio to 0.4 for the rest of the tutorial."
      ],
      "metadata": {
        "id": "mDBGPd9za3mK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "hold = [0.1, 0.2, 0.3, 0.4, 0.5]\n",
        "\n",
        "acc = []\n",
        "for p in hold:\n",
        "  XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=p, random_state=42)\n",
        "\n",
        "  XTrain = XTrain.drop(columns=['Chance of Admit'])\n",
        "  XTest = XTest.drop(columns=['Chance of Admit'])\n",
        "  # Fits logistic regression model to training data.\n",
        "  clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(XTrain, yTrain)\n",
        "\n",
        "  acc.append(clf.score(XTest, yTest))\n",
        "\n",
        "# Plot to compare the model's predicted probabilities vs. the data's given probabilities.\n",
        "plt.plot(hold, acc)\n",
        "\n",
        "plt.xlabel(\"Percent held out\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.title(\"Accuracy Over Hold Out Ratios\")\n",
        "plt.show()\n",
        "\n",
        "XTrain_full, XTest_full, yTrain, yTest = train_test_split(X, y, test_size=p, random_state=42)\n",
        "\n",
        "XTrain = XTrain_full.drop(columns=['Chance of Admit'])\n",
        "XTest = XTest_full.drop(columns=['Chance of Admit'])"
      ],
      "metadata": {
        "id": "8vBACpZ-WmZm",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "567aaa30-93b3-49bd-cf8a-2e5ecd15a895"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZUAAAEWCAYAAACufwpNAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXxU9b3/8debHdkh7GFRQREUWQJ112ptpSooKGhdsLfq9Vprl2t/1Wu369Xa9trq1XrbWmsL3qoEtBa3qmVR22plwo4KBlyysIR9k0DI5/fH+caOMSETMpOZZD7Px2MenPme7XNOwnxyvt8znyMzwznnnEuGFukOwDnnXPPhScU551zSeFJxzjmXNJ5UnHPOJY0nFeecc0njScU551zSeFJxrhmSdJak4kPM/72kOxszplST9B+SHk53HNnOk4qrN0kLJW2T1DbdsaSKpK6Sfilpg6S9klZI+nIj7v+Hkv6vhnaTNKQR9p8r6Q+StkjaI+lNSRfUY/1rJP21jmUWStonabekzZKektQ3we1/Kmma2Y/M7NpEY3Sp4UnF1YukwcDpgAETG3nfrRppP22AvwCDgJOBLsC3gR9L+lYK9tcox5UoSd2BvwL7gRFADnAv8JikS5K8u5vMrCMwBOgI3JPk7btG5knF1dfVwBvA74Hp8TMkDQh/bZaFv3B/ETfvOklvS9ol6S1JY0L7J/7yju+WqfprVNJ3JG0Afiepm6Rnwz62hencuPW7S/qdpNIw/+nQvlLShXHLtQ5/HY+u4RivAgYCl5rZe2Z2wMz+DNwM3CGpc4hpTrXj/x9J94fpLpJ+K2m9pBJJd0pqGeZdI+lvku6VtAX4YX1/CGE7bSXdF461NEzXePUoabSkxeH8zwLaHWLT3wR2A18xsw1m9pGZPQ7cBfxMkcHhZ/dxQgxXHtdKOg74FXByuArZXtexmNl24GlgVNz2vhz3O7NO0r+G9g7AC0C/sP3dkvpVv7qTNFHSKknbQ2zHxc37Tvi57JK0WtI5dcXoEuNJxdXX1cAfwusLknoDhA/MZ4EPgMFAf+CJMO9Sog/Oq4HORFc4WxLcXx+gO9FVw/VEv7O/C+8HAh8Bv4hb/lHgCKK/sHsR/YUNMBO4Mm65LwLrzWxJDfs8F3jBzPZUa3+S6MP45HBsX5TUKe74pwKPhWV/D1QQ/QU+Gvg8EN818xlgHdCb6MP6cNwOnET0QXwiMB74bvWFwpXX00TnpjswG5hyiO2eCzxpZpXV2vOJzvkxhwrKzN4GbgBeN7OOZta1rgOR1AOYDBTGNW8CLiD6nfkycK+kMeHnMgEoDdvvaGal1bZ3DPA48A2gJ/A88IykNpKOBW4CxplZJ+ALwPt1xegS40nFJUzSaUQf5vlmVgCsBb4UZo8H+gHfNrM9ZrbPzKr61K8FfmpmiyxSaGYfJLjbSuAHZlYe/mLeYmZPmtleM9tF9IF8ZoivL9GHzQ1mti1cYbwStvN/REmgc3h/FdGHbE1ygPXVG82sAtgM5IT4FwMXh9lnA3vN7I2QaL8IfCOci01Eye2yuM2VmtkDZlZhZh/VEsfU8Ff2x69q868A7jCzTWZWBvxnOK7qTgJaA/eFczIHWFTLPms9/ri2nEOsW1/3S9pBOK/A16pmmNlzZrY2/M68ArxE1PWaiGnAc2b2spkdIOpWaw+cAhwE2gLDJbU2s/fNbG0SjymreVJx9TEdeMnMNof3j/HPLrABwAfhg7e6AUQJ6HCUmdm+qjeSjpD0a0kfSNoJvAp0DVcKA4CtZrat+kbCX7J/A6ZI6kqUfP5Qyz43A58aMA5dPTlhPkTHf3mY/hL/vEoZRPQhvj4uGfya6MqpSlHdh06+mXWNf1Wb34/oyrDKB6Gtun5AiX2yeuyhknqNxx/XtrmGeYfrZjPrAowEugHxXZkTJL0haWs4h18k8YT2iXMTrrqKgP5mVkh0BfNDYJOkJyTVdN7cYfCk4hIiqT1R986Ziu6I2kDU936ipBOJ/sMOVM2DzkXA0bVsei9Rd1WVPtXmVy+j/e/AscBnzKwzcEZViGE/3UPSqMkMoi6wS4m6ZkpqWe4vwITQdx9vClBONKYEUTfSWWFM52L+mVSKwnI5cQmhs5mNOMRxHY5SogRWZWBoq2490F+Sqi1bm78AkyVV/3yYSnRsa4CqrsHafnb1Oj4zWwHcCTwYxmzaEnU33gP0Dgn1eaKfcyLb/8S5Ccc+ACgJ+3vMzKquvA34SX3idbXzpOISdRFRt8Fwoj78UcBxwGtEYyVvEn14/VhSB0ntJJ0a1n0YuEXS2PCBMURS1X/4pcCXJLWUdB6hK+sQOhGNo2xXdJfSD6pmmNl6ogHc/1U0oN9a0hlx6z4NjAG+TjTGUptHgWJgdhiQbi3pC8D9wA/NbEfYXxmwkGiM570wllAVx0tEg9qdJbWQdLSkuo6tvh4Hviupp6Qc4PtE3XzVvU40vnNzOJbJRN2VtbmX6I6330rqE36WlxON4Xw7dEeVEX1AXxl+dv/CJ/9w2AjkhvGcRM0gGmOaCLQh6qIqAyokTSAal4rffg9JXWrZVj5wvqRzJLUm+mOkHPi7pGMlnR0S1z6i36fq40fuMHlScYmaDvzOzD4MdwRtMLMNRIPkVxD9BXkh0cD0h0QfytMAzGw20djHY8Auog/37mG7Xw/rbQ/bebqOOO4j6hvfTHTF8Odq868CDgDvEA30fqNqRhi7eBI4Eniqth2YWTnwOaK/yv8B7AR+DtxuZv9dbfHHwrKPVWu/muiD8S1gGzCHmruUGuJOIAYsB1YQjfF86guNZrafaBD8GmAr0c/lUMe/BTiN6KaEt4huqvgWcJWZzYpb9DqiW623EN0Y8fe4efOBVcAGSQl1l4U4/wf4Xhgvu5koOWwj6l6cG7fsO0RJdV3oYuxXbVuria5KHyD6XbkQuDDsoy3w49C+gahb8rZEYnR1kz+ky2UTSd8HjjGzK+tc2DlXbxn1pSvnUil0l32Fmu+Qcs4lgXd/uawg6Tqi7qwXzOzVdMfjXHPl3V/OOeeSxq9UnHPOJU1Wj6nk5OTY4MGD0x2Gc841KQUFBZvNrGdN87I6qQwePJhYLJbuMJxzrkmRVGtFBu/+cs45lzSeVJxzziWNJxXnnHNJ40nFOedc0nhScc45lzSeVJxzziWNJxXnnHNJ40nFOdekmRnPLV/Pxp376l7YpZwnFedck7a0aDtffWwxV//2TXaX1/Q0a9eYPKk455q0/FgxbVq1oLBsN994YimVlV4kN508qTjnmqy9+yt4ZlkpF47sx/cvGM5f3t7IPS+tTndYWS2ra38555q2F1ZsYHd5BdPGDWDc4G68s2EX/7twLcf26cSkUf3THV5W8isV51yTNStWxJE5HRg3uBuS+M+JIxh/ZHe+PWc5S4u2pzu8rORJxTnXJL23eQ9vvreVS/NykQRAm1Yt+NWVY+nVqS3Xz4yxYYffEdbYPKk455qk2bEiWgimjMn9RHv3Dm14eHoee8oruP7RGPsOHExThNnJk4pzrsmpOFjJnIJiPntsL3p3bvep+cP6dObeaaNYUbKD/zdnOf7Y9MbjScU51+S8+m4Zm3aVc2negFqX+fyIPtzy+WOZu6yU/124thGjy25+95dzrsmZtaiInI5tOOe4Xodc7sazjmbNxl3c89JqhvbqyOdH9GmkCLNXSq9UJJ0nabWkQkm31jB/kKR5kpZLWigpt9r8zpKKJf0ivD9C0nOS3pG0StKP45a9RlKZpKXhdW0qj805lx6bd5cz7+1NXDy6P61bHvojTBI/mTKSkf278I1ZS3lnw85GijJ7pSypSGoJPAhMAIYDl0saXm2xe4CZZjYSuAO4u9r8/wJerb6OmQ0DRgOnSpoQN2+WmY0Kr4eTdSzOuczxx8UlVFQaUw/R9RWvXeuW/PqqPDq2bcW1M2Js2V2e4gizWyqvVMYDhWa2zsz2A08Ak6otMxyYH6YXxM+XNBboDbxU1WZme81sQZjeDywGPnnrh3Ou2TIz8mNFjB7YlaG9OyW8Xp8u7Xjo6jw27Srn3/6wmP0VlSmMMrulMqn0B4ri3heHtnjLgMlh+mKgk6QekloAPwNuqW3jkroCFwLz4pqnhK60OZJq/DNG0vWSYpJiZWVl9Tsi51xaLSnazrubdjMtwauUeKMGdOW/LxnJm+9t5QdzV/kdYSmS7ru/bgHOlLQEOBMoAQ4CNwLPm1lxTStJagU8DtxvZutC8zPA4NCV9jIwo6Z1zewhM8szs7yePXsm92iccyk1O1ZE+9YtOX9k38Naf9Ko/tx41tE8/uaHzHz9gyRH5yC1d3+VAPF/TuSGto+ZWSnhSkVSR2CKmW2XdDJwuqQbgY5AG0m7zaxqsP8h4F0zuy9uW1viNv0w8NNkH5BzLn2i4pHrOX9kXzq1a33Y27nl88eyZuNu7nj2LYb06sipQ3KSGKVL5ZXKImCopCMltQEuA+bGLyApJ3R1AdwGPAJgZleY2UAzG0x0NTOzKqFIuhPoAnyj2rbi/3SZCLyd/ENyzqXL83HFIxuiRQtx32WjGNKzIzf+YTHvbd6TpAgdpDCpmFkFcBPwItEHfL6ZrZJ0h6SJYbGzgNWS1hANyt91qG2GW45vJxrgX1zt1uGbw23Gy4CbgWuSfUzOufTJX1TEUTkdyBvUrcHb6ti2FQ9Pz6OF4NoZi9i570ASInQAyubBqry8PIvFYukOwzlXh3Vluzn7Z6/wnfOG8W9nHZ207b6xbgtXPvwPTh2SwyPXjKNlCyVt282ZpAIzy6tpXroH6p1zrk6zC4pp2UJMGZPcZ6ScdFQP/nPSCF5ZU8aPX/Ae82TwMi3OuYxWcbCSJwuK+eyxPelVQ/HIhrriM4NYs2EXv3ntPY7t05lLxvpX3xrCr1SccxntlTV1F49sqO9dMJxTh/TgP55aQcEHW1O2n2zgScU5l9GqikeePezQxSMbolXLFjz4pTH069qOf320gJLtH6VsX82dJxXnXMYq21XO/Hc2MXlMbp3FIxuq6xHRw73KD1Ry3YwYe/dXpHR/zZUnFedcxvrjkuJQPLJxxjmG9OrE/ZeP5u0NO7ll9jIqK7P37tjD5UnFOZeRouKRxYwZ2JUhvRIvHtlQnx3Wi9smDOP5FRu4f/67jbbf5sKTinMuIy3+cDuFm3Y3+Bv0h+O6049iyphc7vvLu7ywYn2j778p86TinMtIs2NFHNGmJeeP7Nfo+5bEjyYfz5iBXflW/jJWle5o9BiaKk8qzrmMs6e8gmeWlXL+CX3p2DY9X6dr26olv7pqLF2PaM11M2KU7fKHeyXCk4pzLuM8v2I9e/YfTEvXV7xendrxm6vz2Lp3Pzf8XwHlFQfTGk9T4EnFOZdx8mNFHNWzA2OTUDyyoY7v34WfXTqKgg+2cfsfV/rDvergScU5l1HWlu1m0fvbmJo3ACkzCjyeP7IvN58zlDkFxfz2r++lO5yM5knFOZdRZsei4pGTk1w8sqG+cc5QJhzfhx89/zYLVm9KdzgZy5OKcy5jVBys5MnFxXz22F706pT84pEN0aKF+NnUEzm2T2dufmwJhZt2pzukjORJxTmXMRauLqNsV3mjfYO+vo5oEz3cq23rFlw3M8aOvf5wr+o8qTjnMsasWBE5Hdvy2RQWj2yo/l3b86srx1K8bS9ffWwxFQcr0x1SRvGk4pzLCJt27WP+O5uYMqZ/yotHNlTe4O7cdfEJ/LVwM3c+5w/3ipfSn5yk8yStllQo6dYa5g+SNE/SckkLwzPo4+d3llQs6RdxbWMlrQjbvF/h9hBJ3SW9LOnd8G/670V0ziXsj4tLOFhpKX1uSjJNzRvAV047kt///X0ef/PDdIeTMVKWVCS1BB4EJgDDgcslDa+22D3ATDMbCdwB3F1t/n8Br1Zr+yVwHTA0vM4L7bcC88xsKDAvvHfONQFR8cgixg7qxpBeHdMdTsJumzCMM47pyfeeXsk/1m1JdzgZIZVXKuOBQjNbZ2b7gSeASdWWGQ7MD9ML4udLGgv0Bl6Ka+sLdDazNyz6BtJM4KIwexIwI0zPiGt3zmW4xR9uY23ZHqY1kauUKq1atuCBy0czsMcR/NsfFlO0dW+6Q0q7VCaV/kBR3Pvi0BZvGTA5TF8MdJLUQ1IL4GfALTVss7iWbfY2s6pyohuIEtKnSLpeUkxSrKysrD7H45xLkfxFxaF4ZN90h1JvXdq35rfTx1FxsJLrZsbYXZ7dD/dK92jYLcCZkpYAZwIlwEHgRuB5Mys+1Mq1CVcxNdZSMLOHzCzPzPJ69ux5mGE755JlT3kFzy4v5YKRfemQpuKRDXVkTgcevGIM727azTdnLc3qh3ulMqmUAPHXsrmh7WNmVmpmk81sNHB7aNsOnAzcJOl9onGXqyX9OKyfW8s2N4busapuMv/Kq3NNwHMZUjyyoU4f2pPvnn8cL7+1kZ+/vCbd4aRNKpPKImCopCMltQEuA+bGLyApJ3R1AdwGPAJgZleY2UAzG0x0NTPTzG4N3Vs7JZ0U7vq6GvhTWH8uMD1MT49rd85lsPxFUfHIMQOb/g2b15wymMvGDeAXCwr509KSuldohlKWVMysArgJeBF4G8g3s1WS7pA0MSx2FrBa0hqiMZC7Etj0jcDDQCGwFnghtP8YOFfSu8DnwnvnXAYr3LSb2AfbmJZBxSMbQhJ3TDqe8YO78//mLGdZ0fZ0h9TolM1lnPPy8iwWi6U7DOey1t0vvM3Dr73H67ednXG1vhpiy+5yJv7ib1RUVjL3ptPo3bn5HBuApAIzy6tpXroH6p1zWerAwUqeLCjh7GGZVzyyoXp0bMvD0/PYta+C6x8tYN+B7Hm4lycV51xaLFxdxubd5UxtYt9NSdRxfTtz77RRLCvazq1PLs+ah3t5UnHOpcWsRUX07NSWzx7bfG/t/8KIPtzy+WN4emkpv3plXbrDaRSeVJxzjW7Trn0sWL2JyWP60yrDi0c21Fc/O4QLT+zHT198h7+8tTHd4aRc8/5pOucy0lOheGRz7fqKJ4mfThnJ8f268PUnlrB6w650h5RSnlScc42qqnhk3qBuHN2z6RSPbIj2bVrym6vzOKJtK66duYite/anO6SU8aTinGtUBR9sY13ZHqY28W/Q11efLu146KqxbNxZzo1/KOBAM324lycV51yjyo8V0aFNS84/oekVj2yo0QO78dMpI3lj3VZ+OHdVusNJiaZZvc051yTtLq/g2eXruXBkvyZbPLKhLhrdn3c27OJXr6xlWJ9OXHXy4HSHlFR+peKcazTPLS9l7/6DWdf1Vd23v3As5wzrxQ+feYu/F25OdzhJ5UnFOddo8mPFHN2zA2MGdk13KGnVsoW477JRHN2zA//2h8W8v3lPukNKGk8qzrlGUbhpFwUfbGPauOZRPLKhOrVrzcNXj0OCa2fG2LnvQLpDSgpPKs65RjE7VkyrFuLi0bl1L5wlBvY4gv+9Ygzvb97D1x9fwsFm8HAvTyrOuZQ7cLCSJxcXc/awXvTs1Dbd4WSUU47O4YcTR7BgdRk//fM76Q6nwbLz9gvnXKNa8M4mNu/enxXfoD8cV540iNUbdvHrV9dxTO9OTBnbdK/m/ErFOZdy+bGoeORZzbh4ZEN9/8LhnHxUD257agWLP9yW7nAOmycV51xKbdq5jwWry5gyJrfZF49siNYtW/C/V4yhb9d2XD+zgNLtH6U7pMOS0p+wpPMkrZZUKOnWGuYPkjRP0nJJCyXlxrUvlrRU0ipJN4T2TqGt6rVZ0n1h3jWSyuLmXZvKY3POJebJj4tHNt0uncbSrUMbHr46j30HDnL9ozE+2t/0Hu6VsqQiqSXwIDABGA5cLml4tcXuAWaa2UjgDuDu0L4eONnMRgGfAW6V1M/MdpnZqKoX8AHwVNz2ZsXNfzhVx+acS4yZMTtWxPjB3TkqS4pHNtTQ3p24//JRrCrdyS1zljW5h3ul8kplPFBoZuvMbD/wBDCp2jLDgflhekHVfDPbb2blob1tTXFKOgboBbyWgtidc0kQ+2Ab6zbv4VK/SqmXs4f15tbzhvHc8vU8ML8w3eHUSyqTSn+gKO59cWiLtwyYHKYvBjpJ6gEgaYCk5WEbPzGz0mrrXkZ0ZRKfxqeErrQ5kmq8zUTS9ZJikmJlZWWHd2TOuYTMWhSKR47MvuKRDXX9GUcxeXR/fv7yGv68cn26w0lYukfNbgHOlLQEOBMoAQ4CmFlR6BYbAkyX1LvaupcBj8e9fwYYHNZ5GZhR0w7N7CEzyzOzvJ49/U4U51Jld3kFzy1fz4Un9uOINv7thfqSxI8mn8DogV355qxlvFW6M90hJSSVSaUEiL9ayA1tHzOzUjObbGajgdtD2/bqywArgdOr2iSdCLQys4K45bbEdZk9DIxN4rE45+rp2WWlfHTAi0c2RLvWLfn1lWPp0r41182MsXl3ed0rpVkqk8oiYKikIyW1IbqymBu/gKQcSVUx3AY8EtpzJbUP092A04DVcatezievUpAUf309EXg7icfinKun/FgRQ3p1ZPSA7C4e2VC9OrfjN1fnsWVPOTc8WkB5RWbfEZaypGJmFcBNwItEH/D5ZrZK0h2SJobFzgJWS1oD9AbuCu3HAf+QtAx4BbjHzFbEbX4q1ZIKcHO4/XgZcDNwTQoOyzmXgMJNu1j84Xam5XnxyGQ4IbcL91x6IrEPtvG9p1dm9B1hKe3oNLPngeertX0/bnoOMKeG9V4GRh5iu0fV0HYb0dWOcy7N8quKR46pfm+OO1wXjOzHmg27uH9+Icf26cxXTjsy3SHVKN0D9c65ZubAwUqeWlzMOcf1IqejF49Mpm987hi+MKI3dz33Fq+sycy7Vz2pOOeSar4Xj0yZFi3Ez6eO4pjenbjpscWsLdud7pA+xZOKcy6p8hcV0atTW848xm/ZT4UObVvx8PQ82rRswXUzYuzYm1kP9/Kk4pxLmo0797Fg9SamjPXikamU2+0IfnXVWIq27eWmxxdTcbAy3SF9zH/qzrmkeXJxMZWGd301gnGDu3PnRcfz2rubuev5zPkGhX/N1TmXFFHxyGLGH9mdI3M6pDucrDBt3EBWb9jNI397j2F9OjFt3MB0h+RXKs655Fj0/jbe27zHr1Ia2X98cRhnHNOT7z69kkXvb013OHUnFUkXxn3r3TnnajRrUREd27biiyf0SXcoWaVVyxY8cPloBnQ7ghseLaB42960xpNIspgGvCvpp5KGpTog51zTs2vfAZ5fsZ4LT+zrxSPToEv71vxmeh77D1Zy7YwYe8or0hZLnUnFzK4ERgNrgd9Lej2Uj++U8uicc03Cs8vXR8UjvesrbY7u2ZEHvzSGNRt38a38pVRWpqeUS0LdWma2k6icyhNAX6JnnyyW9LUUxuacayLyY0UM7dWRUV48Mq3OOKYnt58/nBdXbeTev6xJSwyJjKlMlPRHYCHQGhhvZhOAE4F/T214zrlM9+7GXSz5cDvTxnnxyEzwL6cOZlreAB6YX8gzy6o/2zD1Eun8nALca2avxjea2V5JX0lNWM65piI/VkSrFuKi0V48MhNI4r8uOp51m3dzy+xlDO7RgRNyuzTa/hPp/voh8GbVG0ntJQ0GMLN5KYnKOdck7K+o5KnFJXzuuN5ePDKDtGnVgl9eOZacjm25bmaMTTv3Ndq+E0kqs4H4GgAHQ5tzLsvNf2cTW/bsZ+q43HSH4qrJ6diW31ydx859B7j+0QL2HWich3slklRamdn+qjdhuk3qQnLONRX5sSJ6d27LGUO9eGQmGt6vMz+fOoqlRdv5j6dWNMrDvRJJKmVxT2pE0iRgc+pCcs41BRt37mPh6k1c4sUjM9p5x/fhW+cew1NLSvj1q+tSvr9EfhNuAP5D0oeSioDvAP+ayMYlnSdptaRCSbfWMH+QpHmSlktaKCk3rn2xpKXhEcE3xK2zMGxzaXj1Cu1tJc0K+/pH1biPcy415hRExSMvHevfTcl0Xzt7CBeM7MtP/vwO897emNJ9JfLlx7VmdhIwHDjOzE4xs8K61pPUEngQmBDWvVzS8GqL3QPMNLORwB3A3aF9PXCymY0CPgPcKqlf3HpXmNmo8NoU2r4CbDOzIcC9wE/qitE5d3ii4pFFfObI7gz24pEZTxL/fcmJHN+vC19/YilrNu5K2b4SumaVdD5wI/AtSd+X9P261gHGA4Vmti6MwzwBTKq2zHBgfpheUDXfzPabWXlob5tgnJOAGWF6DnCO/KZ551Lizfe28v6Wvf4N+iakfZuWPHT1WNq3acm1M2Js27O/7pUOQyJffvwVUf2vrwECLgUGJbDt/kBR3Pvi0BZvGTA5TF8MdJLUI+x3gKTlYRs/MbP4b/H8LnR9fS8ucXy8PzOrAHYAPWo4nuslxSTFysoy8xnPzmW6WbGq4pF90x2Kq4e+Xdrz66vGsmHnPu55aXVK9pHIFcApZnY1UdfSfwInA8ckaf+3AGdKWgKcCZQQ3bKMmRWFbrEhwHRJvcM6V5jZCcDp4XVVfXZoZg+ZWZ6Z5fXs6XesOFdf/ywe2Y/2bVqmOxxXT2MGduP314zjti8el5LtJ5JUqr41szeMaxwgqv9VlxIg/to4N7R9zMxKzWyymY0Gbg9t26svA6wkSiCYWUn4dxfwGFE32yf2J6kV0AXYkkCczrl6eGbZevYdqGTaOO/6aqpOGZJDx7apqSadSFJ5RlJX4L+BxcD7RB/mdVkEDJV0pKQ2wGXA3PgFJOXEPavlNuCR0J4rqX2Y7gacBqyW1EpSTmhvDVxAlHAI254epi8B5ltj3JTtXJbJjxVxTO+OnNiIpT9c03HIVBU+8OeFq4cnJT0LtDOzHXVt2MwqJN0EvAi0BB4xs1WS7gBiZjYXOAu4W5IBrwJfDasfB/wstAu4x8xWSOoAvBgSSkvgL8Bvwjq/BR6VVAhsJUpizrkkWrNxF0uLtvPd84/z4pGuRqrrj3lJS0L3VLOTl5dnsVgs3WE412Tc+exbzHj9fd647Rx6eK2vrCWpwMzyapqXSPfXPElT/PZc57Lb/opKnloSFY/0hOJqk0hS+VeiApLlknZK2iVpZ4rjcs5lmPnvbGTrnv3+3RR3SHUO/5uZPzbYOeE1I2MAABepSURBVMesRUX06dyOM47xW/Fd7epMKpLOqKm9+kO7nHPN14Yd+3hlTRk3njWEli28J9zVLpEblb8dN92O6HshBcDZKYnIOZdxnlwcikfm+XNT3KEl0v11Yfx7SQOA+1IWkXMuo1RWGvmxIk46qjuDenjxSHdoh/MQhGKi75E457LAm+9v5QMvHukSlMiYygNA1ZdZWgCjiL5Z75zLAvmLiujUthUTjvfika5uiYypxH87sAJ43Mz+lqJ4nHMZZOe+Azy/cj2Tx+R68UiXkESSyhxgn5kdhOjhW5KOMLO9qQ3NOZduzywrjYpHeteXS1BC36gH2se9b09Uc8s518zlx4o5tncnRnrxSJegRJJKOzPbXfUmTB+RupCcc5lg9YZdLCvaztRxA7x4pEtYIkllj6QxVW8kjQU+Sl1IzrlMkB8ronVLcfHo6g9sda52iYypfAOYLamUqAx9H6LHCzvnmqn9FZX8cUkJ5w7vTfcObdIdjmtCEvny4yJJw4BjQ9NqMzuQ2rCcc+k07+2oeOSlPkDv6qnO7i9JXwU6mNlKM1sJdJR0Y+pDc86ly6xYEX27tOOMoV480tVPImMq18U/N97MtgHXpS4k51w6rd/xEa+uKeOSsblePNLVWyJJpWX8A7oktQS8k9W5ZurJglA8cqx3fbn6SySp/BmYJekcSecAjwMvJLJxSedJWi2pUNKtNcwfJGmepOWSFkrKjWtfLGmppFWSbgjtR0h6TtI7of3Hcdu6RlJZWGeppGsTidE5909R8chiTj6qBwN7+DcHXP0lcvfXd4DrgRvC++VEd4AdUriieRA4l6gI5SJJc83srbjF7gFmmtkMSWcDdwNXAeuBk82sXFJHYKWkucB24B4zWyCpDdGjjieYWVWSm2VmNyVwTM65Gvzjva18uHUv3zx3aLpDcU1UnVcqZlYJ/AN4n+hZKmcDbyew7fFAoZmtM7P9wBPApGrLDAfmh+kFVfPNbL+ZlYf2tlVxmtleM1tQtQxRYUt/wINzSZIfK6JTOy8e6Q5frUlF0jGSfiDpHeAB4EMAM/usmf0igW33B4ri3heHtnjLgMlh+mKgk6QeYf8DJC0P2/iJmZVWi68rcCFRGZkqU0JX2pzw3Jeajut6STFJsbKysgQOw7nssHPfAZ5fsZ6JJ/ajXWsvHukOz6GuVN4huiq5wMxOM7MHgINJ3v8twJmSlgBnAiVV+zCzIjMbCQwBpkvqXbWSpFZEYzv3m9m60PwMMDis8zIwo6YdmtlDZpZnZnk9e/rtks5Vmbu0lPKKSqaN8wF6d/gOlVQmE41tLJD0mzBIX5/7C0uA+N/O3ND2MTMrNbPJZjYauD20ba++DLASOD2u+SHgXTO7L265LXFdZg8DY+sRq3NZb3asiGF9OnFCfy8e6Q5frUnFzJ42s8uAYUTjHd8Aekn6paTPJ7DtRcBQSUeGQfXLgLnxC0jKkVQVw23AI6E9V1L7MN0NOA1YHd7fCXQJ8cRvK74TeCKJjfs454B3NuxkWfEOpuZ58UjXMIkM1O8xs8fCs+pzgSVEd4TVtV4FcBPwItEHfL6ZrZJ0h6SJYbGzgNWS1gC9gbtC+3HAPyQtA14huuNrRbjl+HaiAf6qW46rbh2+OdxmvAy4GbgmgeN3zgH5i4pp3VJc5MUjXQPJzOpeqpnKy8uzWCxW94LONWPlFQc56UfzOOXoHB68YkzdK7isJ6nAzPJqmpfIlx+dc83YvLc3sW3vAab6AL1LAk8qzmW5WYuK6NelHacNyUl3KK4Z8KTiXBYr3f4Rr77rxSNd8nhScS6LPVlQjBlc4sUjXZJ4UnEuS1VWGvkFRZxytBePdMnjScW5LPXGe1so2voRU/3pji6JPKk4l6XyF0XFI887vs6i484lzJOKc1lox0cHeGHlBiaN8uKRLrk8qTiXheYuC8Uj8wamOxTXzHhScS4LVRWPPL5/53SH4poZTyrOZZm31+9kefEOpo3z4pEu+TypOJdl8mNFtGnZgotGefFIl3yeVJzLIuUVB/njkhLOHdGbbh3apDsc1wx5UnEui7z81ka27z3ANP9uiksRTyrOZZH8WDH9urTjVC8e6VLEk4pzWaJk+0e89m4Zl+QN8OKRLmU8qTiXJaqKR146NjfdobhmLKVJRdJ5klZLKpR0aw3zB0maJ2m5pIXhccFV7VWPC14l6Ya4dcZKWhG2eb/CPZGSukt6WdK74d9uqTw255qSykojP1bEqUN6MKC7F490qZOypCKpJfAgMIHomfKXSxpebbF7gJlmNhK4A7g7tK8HTjazUcBngFsl9QvzfglcBwwNr/NC+63APDMbCswL751zwBvrtlC8zYtHutRL5ZXKeKDQzNaZ2X7gCWBStWWGA/PD9IKq+Wa238zKQ3vbqjgl9QU6m9kbZmbATOCisNwkYEaYnhHX7lzWmxUronO7VnxhhBePdKmVyqTSHyiKe18c2uItAyaH6YuBTpJ6AEgaIGl52MZPzKw0rF9cyzZ7m9n6ML0B6F1TUJKulxSTFCsrKzu8I3OuCdmxt6p4ZH8vHulSLt0D9bcAZ0paApwJlAAHAcysKHSLDQGmS6oxSdQkXMVYLfMeMrM8M8vr2bNngw/AuUw3d1kJ+ysqmTbOu75c6rVK4bZLgPjf4tzQ9rFw9TEZQFJHYIqZba++jKSVwOnA38J2atrmRkl9zWx96CbblMyDca6pyo8Vc1zfzozo58UjXeql8kplETBU0pGS2gCXAXPjF5CUI6kqhtuAR0J7rqT2YbobcBqwOnRv7ZR0Urjr62rgT2H9ucD0MD09rt25rPVW6U5WlOxgWl6uF490jSJlScXMKoCbgBeBt4F8M1sl6Q5JE8NiZwGrJa0hGgO5K7QfB/xD0jLgFeAeM1sR5t0IPAwUAmuBF0L7j4FzJb0LfC68dy6rfVw8crQXj3SNQ9HwQ3bKy8uzWCyW7jCcS4nyioN85kfzOG1IDr/40ph0h+OaEUkFZpZX07x0D9Q751LkpVWheKQP0LtG5EnFuWYqP1ZE/67tOfVoLx7pGo8nFeeaoeJte/lr4WYuGZtLCy8e6RqRJxXnmqEnC6I77S/x4pGukXlSca6Zqaw0ZhcUcerROV480jU6TyrONTOvh+KRl+b5VYprfJ5UnGtmZi3y4pEufTypONeM7Nh7gD+v2sBFo714pEsPTyrONSN/CsUj/bkpLl08qTjXjOTHihjetzPH9++S7lBclvKk4lwzsap0BytLdvo36F1aeVJxrpnIX1REm1YtmDSqX90LO5cinlScawb2HTjI00tL+cKIPnQ9ok26w3FZzJOKc83AS29tZMdHB5jmA/QuzTypONcMzA7FI085uke6Q3FZzpOKc01cVfHIS/O8eKRLP08qzjVxcwqKAS8e6TJDSpOKpPMkrZZUKOnWGuYPkjRP0nJJCyXlhvZRkl6XtCrMmxa3zmuSloZXqaSnQ/tZknbEzft+Ko/NuUxQWWnMjhVz2pAccrt58UiXfq1StWFJLYEHgXOBYmCRpLlm9lbcYvcAM81shqSzgbuBq4C9wNVm9q6kfkCBpBfNbLuZnR63jyeBP8Vt7zUzuyBVx+Rcpvn72i2UbP+I70wYlu5QnANSe6UyHig0s3Vmth94AphUbZnhwPwwvaBqvpmtMbN3w3QpsAnoGb+ipM7A2cDTKTsC5zLcrFgRXdq35vPDe6c7FOeA1CaV/kBR3Pvi0BZvGTA5TF8MdJL0idtXJI0H2gBrq617ETDPzHbGtZ0saZmkFySNaOgBOJfJtu/dz4urNnDRqH5ePNJljHQP1N8CnClpCXAmUAIcrJopqS/wKPBlM6ustu7lwONx7xcDg8zsROABarmCkXS9pJikWFlZWfKOxLlG9qelpVHxSC/L4jJIKpNKCRD/254b2j5mZqVmNtnMRgO3h7bt8HH31nPA7Wb2Rvx6knKIuteei9vWTjPbHaafB1qH5T7BzB4yszwzy+vZs2f12c41GfmxIkb068yIfl480mWOVCaVRcBQSUdKagNcBsyNX0BSjqSqGG4DHgntbYA/Eg3iz6lh25cAz5rZvrht9ZGkMD2e6Ni2JPmYnMsIK0t2sKrUi0e6zJOypGJmFcBNwIvA20C+ma2SdIekiWGxs4DVktYAvYG7QvtU4AzgmrhbhEfFbf4yPtn1BVGiWSlpGXA/cJmZWSqOzbl0y4+F4pEnVh+mdC69lM2fu3l5eRaLxdIdhnP1su/AQcbf9RfOOrYX918+Ot3huCwkqcDM8mqal+6BeudcPb24agM791V415fLSJ5UnGtiZseKye3WnpOP8uKRLvN4UnGuCSnaGopHjh3gxSNdRvKk4lwTMqegGAkuyfPikS4zeVJxrok4WGnMKYiKR/bv2j7d4ThXI08qzjURf1+7mZLtHzHVn+7oMpgnFeeaiFmLiuh6RGs+P8KLR7rM5UnFuSZg+979vLRqIxeN6k/bVl480mUuTyrONQFPLylh/8FK7/pyGc+TinMZzsyYFSvmhP5dGN6vc7rDce6QPKk4l+FWle7k7fU7meq3EbsmwJOKcxlu1qIi2rZqwcRRXjzSZT5PKs5lsH0HDvKnpSWcd3wfurRvne5wnKuTJxXnMtjHxSN9gN41EZ5UnMtg+bEiBnRvz0lePNI1EZ5UnMtQRVv38rfCLV480jUpnlScy1CzQ/HIKWP9ri/XdHhScS4DHaw05sSKOH1oTy8e6ZqUlCYVSedJWi2pUNKtNcwfJGmepOWSFkrKDe2jJL0uaVWYNy1und9Leq/6s+sVuT/sa7mkMak8NudS6W+Fmyndsc+/m+KanJQlFUktgQeBCcBw4HJJw6stdg8w08xGAncAd4f2vcDVZjYCOA+4T1LXuPW+bWajwmtpaJsADA2v64FfpuK4nGsMs2JR8chzh3vxSNe0tErhtscDhWa2DkDSE8Ak4K24ZYYD3wrTC4CnAcxsTdUCZlYqaRPQE9h+iP1NIkpQBrwhqaukvma2PlkHVOWVNWXc+exbdS/o3GF6b/MerjxpkBePdE1OKpNKf6Ao7n0x8JlqyywDJgP/A1wMdJLUw8y2VC0gaTzQBlgbt95dkr4PzANuNbPyWvbXH/hEUpF0PdGVDAMHDjysA+vYthVDe3c8rHWdS8Twfp259vQj0x2Gc/WWyqSSiFuAX0i6BngVKAEOVs2U1Bd4FJhuZpWh+TZgA1GieQj4DlHXWULM7KGwHnl5eXY4QY8d1I2xg8YezqrOOdespTKplADxXwPODW0fM7NSoisVJHUEppjZ9vC+M/AccLuZvRG3TtWVR7mk3xElpoT255xzLrVSeffXImCopCMltQEuA+bGLyApR1JVDLcBj4T2NsAficZI5lRbp2/4V8BFwMoway5wdbgL7CRgRyrGU5xzztUuZVcqZlYh6SbgRaAl8IiZrZJ0BxAzs7nAWcDdkoyo++urYfWpwBlAj9A1BnBNuNPrD5J6AgKWAjeE+c8DXwQKie4e+3Kqjs0551zNFN0slZ3y8vIsFoulOwznnGtSJBWYWV5N8/wb9c4555LGk4pzzrmk8aTinHMuaTypOOecS5qsHqiXVAZ8cJir5wCbkxhOsmRqXJC5sXlc9eNx1U9zjGuQmfWsaUZWJ5WGkBSr7e6HdMrUuCBzY/O46sfjqp9si8u7v5xzziWNJxXnnHNJ40nl8D2U7gBqkalxQebG5nHVj8dVP1kVl4+pOOecSxq/UnHOOZc0nlScc84ljSeVGkg6T9JqSYWSbq1h/hmSFkuqkHRJtXnTJb0bXtMzKK6DkpaG19zq66Y4rm9JekvScknzJA2Km5fO83WouNJ5vm6QtCLs+6+ShsfNuy2st1rSFzIhLkmDJX0Ud75+1ZhxxS03RZJJyotrS9v5qi2udJ8vSddIKovb/7Vx8xr+/9HM/BX3IirTvxY4iujpksuA4dWWGQyMBGYCl8S1dwfWhX+7helu6Y4rzNudxvP1WeCIMP1vwKwMOV81xpUB56tz3PRE4M9henhYvi1wZNhOywyIazCwMl3nKyzXiejxGW8AeZlwvg4RV1rPF3AN8Isa1k3K/0e/Uvm08UChma0zs/3AE8Ck+AXM7H0zWw5UVlv3C8DLZrbVzLYBLwPnZUBcqZRIXAvMbG94+wbRUzkh/eertrhSKZG4dsa97QBU3U0zCXjCzMrN7D2iZweNz4C4UqnOuIL/An4C7ItrS+v5OkRcqZRoXDVJyv9HTyqf1h8ointfHNpSvW6qt91OUkzSG5IuSlJMhxPXV4AXDnPdxooL0ny+JH1V0lrgp8DN9Vk3DXEBHClpiaRXJJ2epJgSikvSGGCAmT1X33XTFBek8XwFU0K37xxJVY9hT8r5SuUz6l1mGWRmJZKOAuZLWmFmaxszAElXAnnAmY2537rUEldaz5eZPQg8KOlLwHeBpI43Ha5a4loPDDSzLZLGAk9LGlHtyiYlFD2O/OdEXToZo4640na+gmeAx82sXNK/AjOAs5O1cb9S+bQSYEDc+9zQlup1U7ptMysJ/64DFgKjGzMuSZ8Dbgcmmll5fdZNQ1xpP19xngCqrpTSfr5qiit0L20J0wVEffrHNFJcnYDjgYWS3gdOAuaGQfF0nq9a40rz+cLMtsT9rj8MjE103YSkYrCoKb+Irt7WEQ3sVQ10jahl2d/z6YH694gGubqF6e4ZEFc3oG2YzgHepYZBxVTFRfSBvBYYWq09refrEHGl+3wNjZu+EIiF6RF8cuB5HckbeG5IXD2r4iAaIC5Jx+99WH4h/xwQT+v5OkRcaT1fQN+46YuBN8J0Uv4/NvggmuML+CKwJnzg3B7a7iD6axZgHFF/4x5gC7Aqbt1/IRoQLAS+nAlxAacAK8Iv2ArgK40c11+AjcDS8JqbIeerxrgy4Hz9D7AqxLQg/kOB6KpqLbAamJAJcQFT4toXAxc2ZlzVll1I+PBO9/mqLa50ny/g7rD/ZeHnOCxu3Qb/f/QyLc4555LGx1Scc84ljScV55xzSeNJxTnnXNJ4UnHOOZc0nlScc84ljScVl7XiKhGvlDRb0hFpiOEsSafUMu+Hkm6p5/bel5STjG3Vsv3B4dv0ztXIk4rLZh+Z2SgzOx7YD9yQyEqSklne6Cyi78U0FYMBTyquVp5UnIu8BgyR1EHSI5LeDAX/JsHHz6CYK2k+ME9SR0m/C88XWS5pSlju85JeV/Rcm9mSOob29yX9Z2hfIWmYpMFEieyb4YqppsKCwyUtlLRO0scFHCVdGWJcKunXklpWX1HS7ZLWSPorcGxNBx2uPObrn8+UGRjaf6+4Z/JI2h0mfwycHvb7zfqeZNf8eVJxWS9ceUwg+vb87cB8MxtP9LyV/5bUISw6hqj8zZnA94AdZnaCmY0kKjqZQ1Rk8XNmNgaIAd+K29Xm0P5L4BYzex/4FXBvuGJ6rYbwhhGVJB8P/EBSa0nHAdOAU81sFHAQuKLaMY0FLgNGEX3Delwth/8AMCMcwx+A++s4XbcCr4V4761jWZeFvEqxy2btJS0N068BvwX+DkyMG39oBwwM0y+b2dYw/TmiD20AzGybpAuIHgz1N0kQ1V56PW5/T4V/C4DJCcb4nEXF/8olbQJ6A+cQFQFcFPbTHthUbb3TgT9aeF6Man965clxsTxKVNLeucPmScVls4/CX/ofU/QpPcXMVldr/wxRTbVDEVHiubyW+VWVYQ+S+P+98rjpqvVEdHVxW4LbOBwVhJ6MUMa9TQr35ZoR7/5y7pNeBL4WkguSait5/zLw1ao3kroRPT3yVElDQlsHSXWVNN9FVCa9PuYBl0jqFfbTXdKgasu8Clwkqb2kTkRVhWvyd/55xXUF0RUbwPv8syT6RKB1A+J1WcSTinOf9F9EH6DLJa0K72tyJ9At3I68DPismZURPZTpcUnLibq+htWxv2eAiw8xUP8pZvYW0djNS2E/LwN9qy2zGJhFVIn2BWBRLZv7GvDlsJ2rgK+H9t8AZ4ZjO5l/XqUtBw5KWuYD9a4mXqXYOedc0viVinPOuaTxpOKccy5pPKk455xLGk8qzjnnksaTinPOuaTxpOKccy5pPKk455xLmv8PmGWkBxVhJsUAAAAASUVORK5CYII=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Note that the random_state parameter is set to 42 so that the shuffling is controlled and we get consistent results.\n"
      ],
      "metadata": {
        "id": "_yvIhNcnYAS3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now, we can get to work and start predicting!"
      ],
      "metadata": {
        "id": "JiEJiM1OWmvh"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Logistic Regression\n",
        "The first algorithm we will go over in this tutorial is Logistic Regression.\n",
        "\n",
        "Logistic regression is used to model relationships between the qualitative target variable and the relevant independent variables. In fact, the logarithmic likelihood of an outcome is a linear combination of the input. In the binary case, the target variable will only be able to take on one of two values, 0 or 1. For this project, 1 represents that the student is likely to be admitted where 0 means they are not.\n",
        "\n",
        "A more detailed explanation of Logistic Regression as a Binary Classifier can be found at:\n",
        "\n",
        "https://towardsdatascience.com/binary-classification-and-logistic-regression-for-beginners-dd6213bf7162\n",
        "\n",
        "To begin with your logistic regression model, you will want to import the following sklearn packages."
      ],
      "metadata": {
        "id": "fvJmKNSpJpe4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import confusion_matrix"
      ],
      "metadata": {
        "id": "0GOCvsFsm7CE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "The LogisticRegression class implements logistic regression using various libraries and solvers. Further, this class applies regularization as a default and can handle both dense and sparse input.\n",
        "\n",
        "Sklearn's confusion_matrix class will assist us in evaluting the accuracy of our models by creating a confusion matrix.\n",
        "\n",
        "Now we can actually train the model. Thanks to sklearn, this process can be done in just one line!"
      ],
      "metadata": {
        "id": "XQHY0ZWdnFj0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Fits logistic regression model to training data.\n",
        "clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(XTrain, yTrain)"
      ],
      "metadata": {
        "id": "Wg5o8lHioGuU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "The variable clf holds our fitted logistic regression model! We use the default solver and increase the max_iterations to 500 because the solver does not converge for a lower number of iterations.\n",
        "\n",
        "To expand on this a bit more, if the solver is returning an error that is barely changing between iterations, this means that the algorithm has reached a solution and converged.\n",
        "\n",
        "However, if the errors are significantly varying, this means that the algorithm did not converge.\n",
        "\n",
        "For more information, see:\n",
        "\n",
        "https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter\n",
        "\n",
        "Let us now observe the performance of this model."
      ],
      "metadata": {
        "id": "0oQfjU6ZoIBa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy: \")\n",
        "print(clf.score(XTest, yTest))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vkz65Wyqpm-e",
        "outputId": "58ecd673-903a-426a-a931-a4a70e692d6c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: \n",
            "0.935\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Amazing! Despite being so easy to implement, the model has almost a 94% accuracy! Pretty good if you ask us, but there's certainly room for improvement. The goal is, of course, to reach as close to 100% accuracy as possible.\n",
        "\n",
        "Let's see if we can get any more insights on our performance using a confusion matrix.\n",
        "\n",
        "To create the matrix itself is quite simple, also only consisting of one line."
      ],
      "metadata": {
        "id": "H-TsEgpDpsu1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Creates confusion matrix based on model predictions.\n",
        "cm = confusion_matrix(yTrain, clf.predict(XTrain))"
      ],
      "metadata": {
        "id": "ePH-tJ9UqizX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sklearn's confusion_matrix creates a matrix for the training data for us, taking in just our target training data as well as our predictions.\n",
        "\n",
        "The following code allows us to visualize this matrix."
      ],
      "metadata": {
        "id": "vVnIx-8SqjUH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Plots confusion matrix.\n",
        "fig, ax = plt.subplots(figsize=(8, 8))\n",
        "ax.imshow(cm)\n",
        "ax.grid(False)\n",
        "ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))\n",
        "ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))\n",
        "ax.set_ylim(1.5, -0.5)\n",
        "for i in range(2):\n",
        "    for j in range(2):\n",
        "        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "mx-AtLEtXed3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 483
        },
        "outputId": "3c930180-bd85-438c-86b6-cb81091a9bd9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now we know the actual number of correct and incorrect predictions for both classes. We can see that the amount of data for class 0 is much less than for class 1. Further, 0's are being predicted as 1's more often than they are classified correctly.\n",
        "\n",
        "A potential improvement on this is using more examples of students who are unlikely to be admitted.\n",
        "\n",
        "\n",
        "If you're still having a hard time understanding the purpose of a confusion matrix, check out Toward Data Science's great overview:\n",
        "\n",
        "https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62"
      ],
      "metadata": {
        "id": "_80TbO2ErKCX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "As a little experiment on Logistic Regression, we wanted to compare the model's predicted 'Chance of getting in' with our true 'Chance of Admission'.\n",
        "\n",
        "Sklearn uses a one-vs-rest approach for a binary scenario like this. According to scikit-learn.org, LogisticRegression's predict_proba function 'calculates the probability of each class assuming it to be positive using the logistic function'. These values are also normalized.\n",
        "\n",
        "Documentation on LogisticRegression's predict_proba function can be found at:\n",
        "\n",
        "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
      ],
      "metadata": {
        "id": "gRYhKouctl3k"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Model's predicted probabilities.\n",
        "PTrain_plt = pd.DataFrame(clf.predict_proba(XTrain), columns = ['Chance not get in','Chance get in'])\n",
        "\n",
        "# True probability.\n",
        "PTrue_plt = XTrain_full['Chance of Admit'].reset_index()\n",
        "\n",
        "\n",
        "indices = range(0, len(PTrue_plt))\n",
        "\n",
        "# Plot to compare the model's predicted probabilities vs. the data's given probabilities.\n",
        "plt.plot(indices, PTrain_plt['Chance get in'], color = \"red\", label = \"Predicted chance of being admitted\")\n",
        "\n",
        "plt.plot(indices, PTrue_plt['Chance of Admit'], color = \"blue\", label = \"True chance of being admitted\")\n",
        "\n",
        "plt.xlabel(\"Example\")\n",
        "plt.ylabel(\"Chance of getting in\")\n",
        "\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "KNiu6_murEJX",
        "outputId": "4ff54b88-8962-4487-f34c-37af5e8ddeee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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YEJDsYlHEcyenR5bQnDuzMBSBBdeeCHvvvsugwYNCozFXrx4MWvXrg3cce7Zs4cvvviCt99+m7POOguv18uRRx7JL37xi0bHf//99xkxYkTgWIccckhYO5YsWRKUU/jxxx/Zt28fb7/9duDuc8KECXTs2DGh75gwYQL5+fnk5+dz2GGHsW3bNoqKipgzZw7PP/88AN9++y1ffPEFnTp1wu/3B2LfAwYM4J///CcAb7zxBn/+858BK6/Svn17nnrqKVatWsXAgQMDbXlYyCiw9evX061bN4499lgApkyZwty5c7kmhlrr0qULw4YNA+Dcc89lzpw5jB8/nnXr1jF27FgA6urqOOKII8J+/owzzgj8D077LVu2zPqfRRh//PFh2xLg6aefZv78+dTW1rJ161bKy8upr6+nW7dudO/ePWCTW0mdfPLJ+Hw++vbtS11dHePHjwegb9++bNq0Ker/GsrixYtZvHgxxx13HAD79u3jiy++YO/evZx++um0bt0agNNOOy2h40bCOIJkYEJDscmBHEEobdq0CTxXVe6//37GjRsXtM8rr7ySNDvq6+t5//33KSgoSNoxAfJd16XX66W2tpY333yTJUuW8N5779G6dWtGjRoVGHvu8/kCQxCd/SOhqkyZMoU//vGPSbUZGg+DFBFUleLiYt57772Yn3f+71j/QyhfffUVd999NytWrKBjx45MnTo1rnH5zvd5PJ6gNvR4PIAna6gAABNcSURBVAnnmVSVG264gUsvvTRo++wU3Sxl368yFzGKIDZZPGooHsaNG8eDDz4YiPV+/vnn7N+/nxEjRrBo0SLq6urYunUrS5cubfTZIUOG8Pbbb/PVV18B8MMPPwBQWFjI3r17A/uddNJJ3H///YHXjnMaMWIEf/3rXwF49dVX2bVrV9zfEYk9e/bQsWNHWrduzWeffcb7778fsw1Gjx7Ngw8+CFh34nv27GH06NE8++yzgdzIDz/8wNdffx30uR49erBp0ya+/PJLAJ566ilGjhwZ8/u++eabQIf/17/+lRNOOIEePXqwY8eOwPaamho++eSTmMdyGDZsGE8//TQAi99/P2xb/vjjj7Rp04b27duzbds2Xn31VQB69uzJpk2b2LBhAwB/+9vf4v7ecPh8vsD1FHotjBs3jscee4x9+/YBsGXLFrZv386IESN44YUXqKioYO/evbz00kvNssHBOIJkYBRBbLJYEcTDRRddRO/evenfvz99+vTh0ksvpba2ltNPP53u3bvTu3dvzjvvPIYOHdros4ceeijz58/njDPOoKSkhMmTJwNw6qmn8vzzzweSxXPmzGHlypX069eP3r17B0YvzZgxg7fffpvi4mKee+45fvrTn8b9HZEYP348tbW19OrVi+uvv54hQ4bEbIP77ruPpUuX0rdvXwYMGEB5eTm9e/fmtttu46STTqJfv36MHTs2kLx1KCgo4PHHH2fSpEn07dsXj8fDZZddFvP7evTowdy5c+nVqxe7du3i8ssvx+/38+yzz/If//EflJSUUFpamtDIqxkzZrB48WL6jBnDM0uWcPjhh1NYWBi0T0lJCccddxw9e/bk7LPPDoSnCgoKmD9/PhMmTKB///6NQmCJcskll9CvXz/OOeccOnXqxLBhw+jTpw/Tp0/npJNO4uyzz2bo0KH07duXiRMnsnfvXvr378/kyZMpKSnh5JNPDoTkmk2k5EEyHsB4YD3wJXB9mPd/D5QDa4HXgaNjHTMrk8XnnWclQv/yl0xbkr388INqXp7qwoVBm8MltQyGVFFZWak1NTWq336r7z76qJb065dpk1JC1iSLRcQLzAXGApuBFSLyoqq6Z9h8CJSp6gERuRy4E4h+K5ONGEUQm44dYd06OOaYTFtiaMF88803/OY3v6G+qgo/8PDDD2fapKwglcniQcCXqroRQEQWAr/CUgAAqKo7oPo+cG4K7UkdxhHER48embbA0MLp3r07H374IWzZAlu3Qr9+mTYpK0hlwPYo4FvX6832tkhcCLwa7g0RuUREVorIyh07diTRxCRhksUGgyGHyYrMnYicC5QBd4V7X1Xnq2qZqpYdeuih6TUuHowiMBhyiyTUGjqYSGVoaAvQxfW6yN4WhIiMAW4ERqpqVQrtSR1GERgMhhwmlYpgBdBdRLqJiB/4LfCiewcROQ54CDhNVbeHOUZuYBSBwZBbGCUQRMocgarWAtOAfwCfAk+r6icicouIOPOi7wLaAs+IyBoReTHC4bIbowhykp07d1JaWkppaSmHH344Rx11VOB1dXV1yr73iSeeYNq0aSk7fnP57LPPKC0t5bjjjgtMnnJom+CyjvPmzQuUpUgHCZevFgkqXz179uygMuTuCqHxku3nNxwpLTGhqq8Ar4Rsu9n1PLtrxsaLUQQ5SadOnQKzd2fOnEnbtm257rrrAu/X1taSl9fyqrC88MILTJw4kZtuuqnZx4pn4limueWWWwLPZ8+ezbnnnhuo5XPHHXfwhz/8IVOmpY2Wd5WnAuMIksI110CYkj/NorQ0sVp2U6dOpaCggA8//JBhw4bRrl27IAfRp08fXn75Zbp27cpf/vIX5syZQ3V1NYMHD+aBBx7AG1JCY8WKFVx99dXs37+f/Px8Xn/9dQC+++47xo8fz4YNGzj99NO58847Aav88YoVK6ioqGDixInMmjULsMoqT5kyhZdeeomamhqeeeYZevbsyb59+7jqqqtYuXIlIsKMGTM488wzWbx4MTNmzKCqqopjjjmGxx9/vNHd/Jo1a7jssss4cOAAxxxzDI899hjvvfces2fPxuv18vrrr4ctmXHttdeyePFiDj/8cBYuXMihhx7Khg0buPLKK9mxYwetW7fm4YcfpmfPnkEOdtSoUQwePJilS5eye/duHn30UYYPH86BAweYOnUq69ato0ePHnz33XfMnTuXsrKyoO+95ZZbeOmll6ioqOD444/noYceQkRYtWoVF1xwAWCV6XB44okneOGFF9i/fz9ffPEF1113HdXV1Tz11FPkezy8cuedHCLC1KlTOeWUU/juu+/47rvvOPHEE+ncuTODBw8OFCQsLi5mwYIFEc/5448/zh//+Ec6dOhASUlJUH2nXCArRg3lPCY0dFCxefNm3n33Xe65556I+3z66acsWrSI5cuXs2bNGrxeLwsWLAjap7q6msmTJ3Pffffx0UcfsWTJksDiK2vWrGHRokV8/PHHLFq0iG+/tUZa33777axcuZK1a9fy1ltvsXbt2sDxOnfuzOrVq7n88ssDq1LdeuuttG/fno8//pi1a9fyi1/8gu+//57bbruNJUuWsHr1asrKysL+L+eddx5/+tOfWLt2LX379mXWrFn88pe/5LLLLuPaa68N6wT2799PWVkZn3zyCSNHjgw4qksuuYT777+fVatWcffdd3PFFVeEbbfa2lo++OADZs+eHfjsAw88QMeOHSkvL+fWW29l1apVYT87bdo0VqxYwbp166ioqODll18G4Pzzz+f+++/no48+avSZdevW8dxzz7FixQpuvPFGWrduzYcffsjQwYP58+LFQbmCf/u3f+PII49k6dKlLF26lP/6r/8K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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As we can see here, the model's predicted probabilities are much more drastic than our true values.\n",
        "\n",
        "Though we are not sure why this is, it is important to note that as long as the predictions are on the correct side of the 0.5 threshold, the model will the classifying correctly in a **binary case**."
      ],
      "metadata": {
        "id": "zut_idVcuifZ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "For more information and examples of using Logistic Regression check out:\n",
        "\n",
        "https://towardsdatascience.com/binary-classification-and-logistic-regression-for-beginners-dd6213bf7162"
      ],
      "metadata": {
        "id": "6ITbaffEvxKF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## KNN\n",
        "\n",
        "The second binary classification algorithm we will implement is K-Nearest Neighbors or KNN. Similar to logistic regression, we will use KNeigborsClassifier from sklearn for this algorithm.\n",
        "\n",
        "\n",
        "\n",
        "KNN is another supervised learning classifier and it uses proximity to make classifications or predictions about the grouping of a particular data point. An important value that needs to be picked for KNN is k, the number of neighbors. Value of k depends on the dataset and different k values result in different accuracies.\n",
        "\n",
        "To figure out the best k value that will result in highest accuracy for our particular dataset, we decided to use a simple for loop. The loop tries out every k value from 1 to 9 and picks the value that produces the highest accuracy.\n",
        "\n",
        "To compute training and test data accuracy, we used the score function again. The parameters are set to XTrain and yTrain for training accuracy and XTest and yTest for test accuracy. After trying all the values, we achieved an accuracy of 93% when k=3. This is slightly lower than our logistic regression accuracy (94%!) but we are still happy with the result!\n",
        "\n",
        "More information on the K-Nearest Neighbors algorithm can be found at:\n",
        "\n",
        "\n",
        "https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4"
      ],
      "metadata": {
        "id": "s5YqZ_RsHW_a"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "\n",
        "best_knn = KNeighborsClassifier()\n",
        "best_acc = 0\n",
        "\n",
        "neighbors = np.arange(1, 10)\n",
        "train_acc = np.empty(len(neighbors))\n",
        "test_acc = np.empty(len(neighbors))\n",
        "\n",
        "# Loop over K values\n",
        "for i, k in enumerate(neighbors):\n",
        "    knn = KNeighborsClassifier(k)\n",
        "    knn.fit(XTrain, yTrain)\n",
        "\n",
        "    # Compute training and test data accuracy\n",
        "    train_acc[i] = knn.score(XTrain, yTrain)\n",
        "    test_acc[i] = knn.score(XTest, yTest)\n",
        "\n",
        "    if (test_acc[i] > best_acc):\n",
        "      best_knn = knn\n",
        "      best_acc = test_acc[i]\n",
        "      best_k = k"
      ],
      "metadata": {
        "id": "_uBHbP9u1HXf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Below is a plot of accuracy vs. number of neighbors that shows the outcomes for each k value from 1 to 9. We can see that for each number of neighbors, both testing accuracy and training accuracy varies by a decent amount.\n",
        "\n",
        "At k=1, KNN closely follows the training data and shows a high training accuracy. In comparison, the test accuracy is pretty low, indicating overfitting. As the number of neighbors increase, the gap between testing and training accuracy decreases, getting rid of overfitting. It is also observable here that the best results are produced when number of neighbors is 3."
      ],
      "metadata": {
        "id": "qwY4Z1MY1JRd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Generate plot\n",
        "plt.plot(neighbors, test_acc, label = 'Testing Accuracy')\n",
        "plt.plot(neighbors, train_acc, label = 'Training Accuracy')\n",
        "\n",
        "plt.legend()\n",
        "plt.xlabel('# Neighbors')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "ec5Z1ifq2DjH",
        "outputId": "a0c8746f-7de1-4ad4-846f-0e9646113ec0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Let's make a confusion matrix once again to get more insight on our performance on KNN. Similar to the confusion matrix for logistic regression, there are a lot more data for class 1 compared to class 0. Also again, 0's are being predicted as 1's more often than they are classified correctly whereas 1's are mostly being predicted correctly."
      ],
      "metadata": {
        "id": "AXsY0cOR5AMl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "cm = confusion_matrix(yTest, best_knn.predict(XTest))\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 8))\n",
        "ax.imshow(cm)\n",
        "ax.grid(False)\n",
        "ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))\n",
        "ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))\n",
        "ax.set_ylim(1.5, -0.5)\n",
        "for i in range(2):\n",
        "    for j in range(2):\n",
        "        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')\n",
        "plt.show()\n",
        "\n",
        "print(\"Accuracy: \")\n",
        "best_knn.score(XTest, yTest)"
      ],
      "metadata": {
        "id": "yYTYwRMIGeOi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 519
        },
        "outputId": "d505d702-2e64-4c35-d5a9-5d67fbfd8b09"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: \n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.925"
            ]
          },
          "metadata": {},
          "execution_count": 98
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "best_k"
      ],
      "metadata": {
        "id": "GqzoZmKI2YEq",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8c24265b-3c82-4feb-ac74-5f15debd046e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3"
            ]
          },
          "metadata": {},
          "execution_count": 99
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Decision Tree Classifier\n",
        "We also decided to apply the Decision Tree Classification model to our data set as well, another form of binary classification. In order to use this model we imported the DecisionTreeClassifier object also found in the sklearn libraries, specifically sklearn.tree.\n",
        "After creating the classifier object and fitting it to our training data, we created our predictions by applying this model to the allocated testing data.\n",
        "\n",
        "A more detailed decsription of Decision Tree Classification can be found at:\n",
        "\n",
        "https://towardsdatascience.com/decision-tree-classifier-explained-in-real-life-picking-a-vacation-destination-6226b2b60575"
      ],
      "metadata": {
        "id": "Si9MI6BI6Cnn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "import seaborn as sns\n",
        "\n",
        "\n",
        "# create a classifier object\n",
        "DTC = DecisionTreeClassifier()\n",
        "# fit the model\n",
        "DTC.fit(XTrain,yTrain)\n",
        "# predict the values for test data\n",
        "pred2 = DTC.predict(XTest)\n",
        "\n",
        "# Regression Score of the model\n",
        "print('Score For Train Data : {}'.format(DTC.score(XTrain,yTrain)))\n",
        "print('Score For Test Data : {}'.format(DTC.score(XTest,yTest)))\n",
        "\n",
        "print('The mean absolute error:', metrics.mean_absolute_error(yTest, pred2))\n",
        "print('The mean squared error:', metrics.mean_squared_error(yTest, pred2))\n",
        "print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(yTest, pred2)))\n",
        "print('\\n')\n",
        "\n",
        "# Plot showacasing the how well model fitted on testing data\n",
        "# sns.scatterplot(x=yTest, y=pred2)\n",
        "\n",
        "\n",
        "indices = range(0, len(pred2))\n",
        "\n",
        "# Plot to compare the model's predicted probabilities vs. the data's given probabilities.\n",
        "\n",
        "plt.scatter(indices, pred2, color = \"red\", label = \"Predicted chance of being admitted\", alpha = 0.3)\n",
        "plt.scatter(indices, yTest, color = \"blue\", label = \"True chance of being admitted\", alpha = 0.3)\n",
        "plt.xlabel('examples')\n",
        "plt.ylabel('predictions')\n",
        "plt.title('Actual test data vs Model predictions ')\n",
        "plt.legend()\n",
        "plt.show()\n",
        "\n"
      ],
      "metadata": {
        "id": "WUQVSD_I6KIu",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 420
        },
        "outputId": "91639e9b-8729-4d81-ba6b-ff595ae0a7d0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Score For Train Data : 1.0\n",
            "Score For Test Data : 0.895\n",
            "The mean absolute error: 0.105\n",
            "The mean squared error: 0.105\n",
            "The root mean squared error: 0.324037034920393\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXxU9bn48c8zM5nsJCFEBYKAFlTCJptwrYhVFJeKWnetaLVutdu91drbXqV2uW311yqupd6qtVoX6lavVq8W614DiBturBLWJGQly2Rmnt8f58w4CZNkEjIJeJ7365VXzpzt+5zvWZ45y3yPqCrGGGO8yzfQARhjjBlYlgiMMcbjLBEYY4zHWSIwxhiPs0RgjDEeZ4nAGGM8zhKB2YWILBSRPw90HMmIyHoROWag4xhoIvKSiFyS4rgqIl9Kd0ydlB2PU0TOE5HnezmfZ0VkQd9GZ2IsEeyB3J2nRkQyUxz/QhF5Nd1xuWXNEZGKPppXygezXs5/wA6ACTEsdOP4bof+33X7Lxyg0Pqdqj6gqsd2N16yLyKqeryq3pe+6LzNEsEeRkRGAUcACpw8oMGYvvIJcEGHfgvc/nsNEQkMdAwmPSwR7HkuAN4E7sU5WMSJyAgReUxEKkWkWkRuE5FDgLuAWSLSKCK17rjtvm13PGsQkVtEZKOI1IvIchE5orvARCQXeBYY5pbVKCLDRMQnIteKyBo3rkdEZLA7TZaI/NntXysi5SKyr4j8Aifh3ebO57ZOyvy6iGxwp/9xh2EzROQNd75b3PoIusNedkd7x53/WSJSJCJPu/VX43aXdlLuD0VkSYd+t4jIooT6XCsiDSKyTkTO66LqyoEcESlzpy0Dstz+ifP/poisFpEdIvKUiAxLGDZXRD4SkTq3rqTDtN8QkQ/d5XpOREZ2EU/idC+JyH+LyFvutvBkwrob5Z61XCwinwH/6K6sruJMsg2Wicj/ucu7TUT+U0TmAf8JnOWut3cS4oxdYvKJyE/c7WK7iPxJRAo6xLxARD4TkarE7cbdZpa5y7pNRH6bSj194amq/e1Bf8Bq4EpgKtAG7Ov29wPvAL8DcnEOJF92h10IvNphPi8BlyR8bjcOcD5QDASA/wC2AlnusIXAnzuJbw5Q0aHfd3GSVymQCfwe+Is77DLgb0COuwxTgUHJYkxS1jigEZjtzve3QBg4xh0+FZjpLsMo4EPgewnTK/ClhM/FwNfcWPKBR4EnOil7JNAE5CfU/xa3vFygHjjIHTYUKOtkPguBP+Mc3H7t9vsN8CO3/0K331eAKmCKu6y3Ai+7w4YADcDpQAbwfbceLnGHz3e3m0PcuvgJ8Hpn9ZBkO9kEjHeX66+xde/WqQJ/codld1VWCnFeiLsNuvW/BWfby3I/H9bZ9pe4rQDfcGM4AMgDHgPu7xDzH9x4JwGtwCHu8DeAr7vdecDMgd7n94S/AQ/A/hJWBnwZ5+A/xP38EfB9t3sWUAkEkkwX38ES+sV3nM7G6TB+DTDJ7d5lR0wYbw67JoIPgaMTPg91lyPg7rSvAxOTzKtdjEmGXwc8lPA5FwjhJoIk438PeDzhc6cHQHf4ZKCmi+GvAhe43XOBNQlx1OIklexu1ulCnAP+/sBn7gHyM2AE7RPB/wC/SZguz63DUbhniQnDBKhIODA+C1ycMNyHk8RGdlcP7jr4VcLncW4d+/n8oHpAwvBOy0ohzvg2CJwDvN1VnXW2rQAvAlcmDDsoYXuLxVyaMPwt4Gy3+2Xgp7j7mP05f3ZpaM+yAHheVavczw/y+eWhEcAGVQ33RUEi8gP39L7OvZxUgPONrjdGAo+7l2hqcRJDBNgXuB94DnhIRDaLyG9EJCPF+Q4DNsY+qOpOoDphGca6l3e2ikg98MuulkFEckTk9+4lhXqcg0KhiPg7meRBnAMWwLnu51gcZwGXA1tE5H9F5OCuFkRVP8P5FvtL4FNV3dhhlGHAhoTxG91lHZ6kHjTxM07935JQ/ztwDsLDu4opQeK8NuAkqyGdDO+qrO7iTDQCWJNifB21qyu3O4CzvcVsTehuwkmsABcDY4GP3MuUJ/Uyhi8USwR7CBHJBs4EjnQPbFtxTq0nicgknB1qf0l+wy5ZE7I7cS6BxOyXUNYRwDVueUWqWgjU0eG6cyeSlbUROF5VCxP+slR1k6q2qepPVXUc8G/ASXx+47S7pm+34BwwYnHn4FzeibkT56xpjKoOwrn80tUy/AfOt8fD3PFnx2bdyfiPAnPc+win4iYCAFV9TlXn4pz9fIRzKaI7f3Jj+FOSYZtxDrJOQM79mGKcyzYd60ESP+PU/2Ud6j9bVV9PISY6zGt/nG/XVQn9EtdTV2V1Fycd5nNAJ8O62y7a1ZUbcxjY1s10qOqnqnoOsA/wa2CJW9eeZolgz3EKzrfocTiXLCbjXId9BefA+RbOjvYrEckV5ybs4e6024DS2I1S10rgNPdb8JdwvgnF5OPsOJVAQESuAwalGOc2oDh2c851F/CL2E1DESkRkflu91EiMsH91l2Pc5CJJsyrs4MBwBLgJBH5srtsN9B+m81359nofiO/IkmsB3QYvxmodW+IXt/VgqpqJc4liXuAdar6obtM+4rIfPcA0opzHyPa6Yw+9zBwLPBIkmF/AS4SkcniPDb8S+Bfqroe+F+gTEROc78IfIeExI5T/z+Sz29GF4jIGSnEE3O+iIxzE+0NwBJVjXQybldldRdnoqeBoSLyPRHJFJF8ETnMHbYNGCUinR2f/gJ8X0RGi0geTl09nMrZsoicLyIlqhrFubwHqa27LzRLBHuOBcA9qvqZqm6N/QG3AefhfGv9KvAlnGvMFTiXJ8B5muMDYKuIxL7J/Q7nWu824D7ggYSyngP+jvP44gaghc5P4dtR1Y9wdsS17uWBYcAtwFPA8yLSgHPjOLZT74dzQK/HuWT0T5zLRbjTnS7O0yeLkpT1AfAtnG/iW3DuYyT+huEHOJdsGnC+kT/cYRYLgfvcOM8Ebsa5gVjlxvj3FBb5QeAYEs4GcPabf8f5ZroDOJJdk9AuVLVZVV9Q1eYkw14A/gvnZu0W4EDgbHdYFXAG8Cucy0VjgNcSpn0c59vtQ+4lr/eB41NYtpj7cZ5S24pz4/Y7XSxDp2V1F2eH+TTg3Hf5qlvup8BR7uBH3f/VIrIiyeR/dGN+GViHs/1+O6UlhXnAByLSiLP9nZ1sfXiNuDdQjDEeJCIv4dyYvXugYzEDx84IjDHG4ywRGGOMx9mlIWOM8Tg7IzDGGI/b6xqRGjJkiI4aNWqgwzDGmL3K8uXLq1S1JNmwvS4RjBo1imXLlg10GMYYs1cRkQ2dDbNLQ8YY43GWCIwxxuMsERhjjMdZIjDGGI+zRGCMMR6XtqeGROSPOE0Ob1fV8UmGC06jTyfgtBd+oaoma2Bqt1WUb6Z8yQY+XtnMho1CY3MARRmU0cLI3B0MDtYjWdlEi0so2dfH9NFVsGMHz75RwIrN+6ECB+RuZ3BmE1XRwWyQUTRKHtoSQhrrIRJ15pcdZeT+yuASHzVVEdZt8NHQ7ENQwIei7boHZUcpGBShtt4Xj6mzcWPduRktn8fSksuGxiE0RrKdcfwByMtHg0GkrRUad6KqDBo+iIJRRdSur6Vxcz0abuu2nK7KnzZ4PfOGvQeqPLt5Iit2jKIhnNnl8uRmRzhgZJTBJX6qtkXarYdUyk+sWxF6NY8eL+fQrUwaWcvKDYWs2LwfDeHMvisnYV0Nym6jQGuprYdGCpx+vkZG6noGU42IEFUfJYUhSsfkUBEdxsdr/J0uf2fbYW+2t950dyyn023W70fy8yAzc5d9qVfrR3PbzW9QqIqRmdsYHKynJpTLusZ9nO00Yb+P9W+3j3fYrwaq3jpbr2MnZTH99JGUTo+/yXS3pe2XxSIyG6d53j91kghOwGkx8ASclipvUdXDOo7X0bRp07Qnj49WlG/myZs+JRJW/rXMz8aafMIRRVSRSBslmQ0EApATaGH2sLVkBZV1jcXUt2SxfWcORVJLY2uQd3cewJDsnWTkZrCjPkjYH6RVM6gJ5VAk9WT52xCJUpLTTCjip6o5h9GDa/hsxyC2thRSFGwEhZq2PIqCjWT5I7RGfLRFA2QGwgSI0BoJxIcnjhvrHppVw/6BzaxtHsaQzHoyJMyO1jzCvgxayaImmk9RZjNolJrWHIryImRlQmtziLaQj8xMJeCH1p1hJ+5Oyumq/HGZawmFYN+cBgC2N+URCAprI/tTH8pKujxDs+vYv7CBtTUFTh0GouxoyiUc0W6XOVZXsboNZIBGlXBUejSPHi9nzgZq2nKpDeVSlLmTAv9OVu0cydZQ0e6Xgy++rrICEVqjPtqiQTIzIRBtRSJhJByiZFCIQMtOckK1zN7nI3YWDue1dcMYV7yVT5pGsLFx8C7Ln1hXidvh9oY86lszerS99aa743ad529hX9m+6zYbyKQ1I5+apgyKgk3g88X3JTTa8/WT2cqqti+xtSGHomATWRlRZ//2VTv10DqI0Tlb2B4eQqgNSjIbCGmAqtZBTMhfh0TV2cc77Ffi9/d4P+2LeutYTmy9jireyYypYfwBH/N/MKZHyUBElqvqtGTD0nZGoKovi8ioLkaZj5MkFHhTRApFZKiqbunLOMqXbKCwOMAHb4doaA0wOKeZzQ15EAkzLKuJiuYhDM2uZf+iMGu25TFrxCaqmkeyekcRB5XUkNPUREWogMGZTWxvK4Q6YXhuHZt3ZlAXDlKU1UJdSy5+Xz3D8lqoqB8ECkPymtlQU0BE/RQFm6hrc959Eev2+xrx+YT6UDYlgQbCBKgL5yYdN9bdFhbWR/ZzYmktBIHhWTvY3DKYOnIoymqlriUTUKe7NRd/ZhQfIerDmZQEmwlHhLpoTpfldFV+rWSzf84OVjU473w/KH8Ln+0sZlDGThp92UmXp039rK/NZ3BOC9t35oHC8IJGNjfkdbvMsbqK1e3QQc4OtKU+r0fz6PFyhvMIkUFV2yByMtqoQwhHpW/K8Ut8Xfl9LfiA+rYsSnIjhJtD0KYMy2ulom4QQwOt7J/TzJrmYdAqFOeHeLdqOG344tty4vIn1lXidrhvXjONbcEebW+96e64XQ/y72R9c5JtNpxLXUvA3U6z3W3W2ZeIRHu+fnw5hFvC8fn5tZ5h+S1U1O8DkQhDMhvZ0LIv+/qrIStIRfMQEBiS2UhdaxYoSfcrfL4e76d9UW8dy4mt1/qWDLZuCVM2OUD5kg19dlYwkPcIhtO+DfwKOnm1nohcKiLLRGRZZWVljwqp3NRGXqGfukYhHPYRDESJqI9IRAgGIjRHg4SjfrKCYWeDiEQIhf00tGWRnRGGqNIUySIvo4XmaCbNbUGCGRCJQCgSJMvfRigaJBL1O/OLZNIcySQv2EZjWw4R9TnjaIBQNBDvjkT9RNRHWP3x7sThybojUaExnOPGEqQ5EiTojxKJCqFopjN+JMOJKxAmFPETUSES9RGO+IlEnOWOj9tJOV2V3xTOJCsYoSGSQ0Mkh+yMCE2RLIKEOl2eiPpobMshL9gWr5/Yekil/MS6DYf9hKO+Hs+jx8sZySIc9dMW9ROO+mmKZBKJSh+V8/m6ikR8zvpRX3xdRcI+ghlKczi2bUaoC+dS1xykMLuV6lA+4TBJl7+z7TA2bk+2t950d9yug7Ql32bVRyic4W6nGe32pV6tH80mEiE+v0jY5+7fmTRHguRltNAYziEYbY3v97H+TeHMhH28/X7Vm/20L+qts/Uajvioq/eRV+inclNb74++HewVvyxW1cXAYnAuDfVk2pLhGTTWRijIUwKBKKGwD79Ewa+Ewn6yfSECvggtoQAFmfXgdyo8P6OF5rYAOT4hx99CY1sB2b5W8AuhNvD7IaghWiIZBH0h/D4ngWT7W0GhMZRBXkYTkajfGUfCIMS7/b4IIAQkEu8O+sJJx411+30h8qSJxrYssn0hEAhFfPh9SpBWZ3x/G6C0hAME/RH8ouCLEvBH8PujIELQ19plOV2VnxNopSXkJ9/fBEBzm58cfwshcvBLNOny+H1R8jKaaAxlxOsnth66W+ZYXcXqNhCIgEK2v7VH8+jxcvpbaNUMMnwRAr4Imb42/D6lJRLc/XJ8kfi68vudl2MFJBpfVwSihNqE7EBs2/RTENgJQR+1zfkUBxtoI5B0+TvbDmPj9mR76013x+06RAZ5gSTbrEQJBtrc7dTdZt19CY32fP1IM35/MD4/fyDq7t+t4I/Q2JZFXqCJkC8T3P0ewe1f59RVW8Eu+xW92E/7ot46lhNbrwF/lIJBURprI5QMT/XV390byESwifbvMy11+/Wp6aeP5MmbPmW/YcKGzWE21uQTIIz4laqWHEqzqgj4oaohwOxhjdQziCHZjQSLw2xvyCEqORQGm/ls51D2ya517xFkEwgIBf6Qcy3eV0+GRKnaGaQ0v965Jtn0+T2CqtaEewQh55pfhkRpjfgYFHBejhQgTEFgZ3x44rix7qFZrewf2Mra5mHsk1nrXMtsySPgi1BAEzUtCfcIWnIoymslA+ddioMCrSBCwA8Fvqb29wi6KLNj+YXBZqqashmX77wobHtTHgWZzayNlBCNatLlSbxHsE9OY/weQSrLHKurWN0GAs49guEF9T2aR4+XM9BITVsuQzLqyfSFKPDvJOArpqoX9bZLN774usrwRWmN+hiU0QJhCPgVQajamUlpQQOBljBVTdnM3mcDOwuHs2ZdkIlDNrn3CLJ3Wf7OtsPtDXmdrp++qrdk23V9JJdRmUm22UCYgqwwNU2Zn98jcPclJNrz9RNtIJBVTFWDM7+MDD9VTVmUBrfveo+gBUqzquL3CIbnb0Oi6uzjHfYrEXq8n/ZFvXW2XkcV72S/oVBbHebIi8b02XEyrc1Qu/cInu7kZvGJwFV8frN4karO6G6ePb1ZDPbUkD01ZE8N2VND9tRQVzeL0/nU0F+AOcAQnPfmXg9kAKjqXe7jo7fhvEO0CbhIVbs9wvcmERhjjNcN1FND53QzXHFeTG6MMWYA2S+LjTHG4ywRGGOMx1kiMMYYj7NEYIwxHmeJwBhjPM4SgTHGeJwlAmOM8ThLBMYY43GWCIwxxuMsERhjjMdZIjDGGI+zRGCMMR5nicAYYzzOEoExxnicJQJjjPE4SwTGGONxlgiMMcbjLBEYY4zHWSIwxhiPs0RgjDEeZ4nAGGM8zhKBMcZ4nCUCY4zxOEsExhjjcZYIjDHG4ywRGGOMx1kiMMYYj7NEYIwxHmeJwBhjPM4SgTHGeJwlAmOM8bi0JgIRmSciH4vIahG5Nsnw/UVkqYi8LSLvisgJ6YzHGGPMrtKWCETED9wOHA+MA84RkXEdRvsJ8IiqHgqcDdyRrniMMcYkl84zghnAalVdq6oh4CFgfodxFBjkdhcAm9MYjzHGmCTSmQiGAxsTPle4/RItBM4XkQrgGeDbyWYkIpeKyDIRWVZZWZmOWI0xxrMG+mbxOcC9qloKnADcLyK7xKSqi1V1mqpOKykp6fcgjTHmiyydiWATMCLhc6nbL9HFwCMAqvoGkAUMSWNMxhhjOkhnIigHxojIaBEJ4twMfqrDOJ8BRwOIyCE4icCu/RhjTD9KWyJQ1TBwFfAc8CHO00EfiMgNInKyO9p/AN8UkXeAvwAXqqqmKyZjjDG7CqRz5qr6DM5N4MR+1yV0rwIOT2cMxhhjujbQN4uNMcYMMEsExhjjcZYIjDHG4ywRGGOMx1kiMMYYj7NEYIwxHmeJwBhjPM4SgTHGeJwlAmOM8ThLBMYY43GWCIwxxuMsERhjjMdZIjDGGI+zRGCMMR5nicAYYzzOEoExxnicJQJjjPE4SwTGGONxlgiMMcbjLBEYY4zHWSIwxhiPs0RgjDEeZ4nAGGM8zhKBMcZ4nCUCY4zxOEsExhjjcZYIjDHG4ywRGGOMx1kiMMYYj0spEYjId0VkkDj+R0RWiMix6Q7OGGNM+qV6RvANVa0HjgWKgK8Dv+puIhGZJyIfi8hqEbm2k3HOFJFVIvKBiDyYcuTGGGP6RCDF8cT9fwJwv6p+ICLS5QQifuB2YC5QAZSLyFOquiphnDHAj4DDVbVGRPbp8RIYY4zZLameESwXkedxEsFzIpIPRLuZZgawWlXXqmoIeAiY32GcbwK3q2oNgKpuTz10Y4wxfSHVRHAxcC0wXVWbgCBwUTfTDAc2JnyucPslGguMFZHXRORNEZmXbEYicqmILBORZZWVlSmGbIwxJhUpXRpS1aiIbAPGiUiql5NSLX8MMAcoBV4WkQmqWtuh/MXAYoBp06ZpH5ZvjDGel9JBXUR+DZwFrAIibm8FXu5isk3AiITPpW6/RBXAv1S1DVgnIp/gJIbyVOIyxhiz+1L9dn8KcJCqtvZg3uXAGBEZjZMAzgbO7TDOE8A5wD0iMgTnUtHaHpRhjDFmN6WaCNYCGUDKiUBVwyJyFfAc4Af+6D5tdAOwTFWfcocdKyKxM42rVbW6R0tg9mptbW1UVFTQ0tIy0KEY84WQlZVFaWkpGRkZKU8jqt1fcheRvwKTgBdJSAaq+p1exLlbpk2bpsuWLevvYk2arFu3jvz8fIqLi+nmiWRjTDdUlerqahoaGhg9enS7YSKyXFWnJZsu1TOCp9w/Y/pUS0sLo0aNsiRgTB8QEYqLi+np05WpPjV0n4gEca7hA3zs3uA1ZrdZEjCm7/Rmf0q1raE5wKc4vxS+A/hERGb3uDRj9kB+v5/Jkyczfvx4zjjjDJqamno9rwsvvJAlS5YAcMkll7Bq1apOx33ppZd4/fXXe1zGqFGjqKqqSmnchQsXctNNN/W4jP7yyiuvUFZWxuTJk2lubo73X79+PePHj+/RvK677jpeeOGFvg6xUy+99BInnXRSj6ZJ3CZ++ctfxvvX1tZyxx139DiGvlq/qf6g7P8Bx6rqkao6GzgO+N1ul27MHiA7O5uVK1fy/vvvEwwGueuuu9oND4fDvZrv3Xffzbhx4zod3ttE8EXywAMP8KMf/YiVK1eSnZ29W/O64YYbOOaYY/oosvRI3Cb6IhH0lVQTQYaqfhz7oKqf4DxFZEz/qqiAxx+HxYud/xUVfTr7I444gtWrV/PSSy9xxBFHcPLJJzNu3DgikQhXX30106dPZ+LEifz+978HnJtzV111FQcddBDHHHMM27d/3krKnDlziD3Y8Pe//50pU6YwadIkjj76aNavX89dd93F7373OyZPnswrr7xCZWUlX/va15g+fTrTp0/ntddeA6C6uppjjz2WsrIyLrnkEjp7wKNjGTGrVq1izpw5HHDAASxatCje/5RTTmHq1KmUlZWxePHieP+8vDx+/OMfM2nSJGbOnMm2bdsA2LZtG6eeeiqTJk1i0qRJ8ST25z//mRkzZjB58mQuu+wyIpEIHb344osceuihTJgwgW984xu0trZy991388gjj/Bf//VfnHfeebtMEw6HOe+88zjkkEM4/fTT42dqy5cv58gjj2Tq1Kkcd9xxbNmyBWh/NjZq1Ciuv/56pkyZwoQJE/joo48AqKysZO7cufG6HDlyZNKzqyuuuIJp06ZRVlbG9ddf366ODz74YKZMmcJjjz0W779w4UIWLFjAEUccwciRI3nssce45pprmDBhAvPmzaOtra3dNnHttdfS3NzM5MmTOe+887j22mtZs2YNkydP5uqrrwbgxhtvjG9viTH84he/YOzYsXz5y1/m44/jh+Xdo6rd/gF/BO7G+QXwHOAPOI+DpjR9X/5NnTpVzRfHqlWrUh9540bV225T/fOfVZ94wvl/221O/92Qm5urqqptbW168skn6x133KFLly7VnJwcXbt2raqq/v73v9ef/exnqqra0tKiU6dO1bVr1+pf//pXPeaYYzQcDuumTZu0oKBAH330UVVVPfLII7W8vFy3b9+upaWl8XlVV1erqur111+vN954YzyOc845R1955RVVVd2wYYMefPDBqqr67W9/W3/605+qqurTTz+tgFZWVrZbhq7KmDVrlra0tGhlZaUOHjxYQ6FQu3Gampq0rKxMq6qqVFUV0KeeekpVVa+++ur4cp955pn6u9/9TlVVw+Gw1tbW6qpVq/Skk06Kz/OKK67Q++67r11szc3NWlpaqh9//LGqqn7961+Pz2fBggXx+kq0bt06BfTVV19VVdWLLrpIb7zxRg2FQjpr1izdvn27qqo+9NBDetFFF+0yr5EjR+qiRYtUVfX222/Xiy++WFVVv/Wtb+kvf/lLVVV99tlnk9ZlYt2Ew2E98sgj9Z133okvxyeffKLRaFTPOOMMPfHEE+P1fPjhh2soFNKVK1dqdna2PvPMM6qqesopp+jjjz+uqp9vE6qfb3ex5S0rK4t/fu655/Sb3/ymRqNRjUQieuKJJ+o///lPXbZsmY4fP1537typdXV1euCBB7bbhmKS7Vc4j+0nPa6m+tTQFcC3gNjjoq/g3Cswpv+Ul0NhIQwa5HyO/S8vh9LSXs829s0MnDOCiy++mNdff50ZM2bEH8F7/vnneffdd+PfOOvq6vj00095+eWXOeecc/D7/QwbNoyvfOUru8z/zTffZPbs2fF5DR48OGkcL7zwQrt7CvX19TQ2NvLyyy/Hv32eeOKJFBUV9aiME088kczMTDIzM9lnn33Ytm0bpaWlLFq0iMcffxyAjRs38umnn1JcXEwwGIxf+546dSr/93//B8A//vEP/vSnPwHOfZWCggLuv/9+li9fzvTp0+N1uc8+7RsR/vjjjxk9ejRjxzrPmixYsIDbb7+d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DiPZj0ZvuOEGpk+fzvjx47n00kvjzXEvX7483jT27bffHh//3nvv5ZRTTmHu3LmMGjWK2267jd/+9rcceuihzJw5M94MR6z56kWLFrF582aOOuoojjrqqF2aiu5qnd9zzz2MHTuWGTNmxJsP35t019bQNe7/W0VkUce//gnRGEdFBTz5JDQ1OW3PNTU5n/s6GVRUVPD666/z29923sbihx9+yMMPP8xrr73GypUr8fv9PPDAA+3GCYVCnHXWWdxyyy288847vPDCC/GXr6xcuZKHH36Y9957j4cffpiNGzcCTlvzy5Yt49133+Wf//wn7777+c93hgwZwooVK7jiiiviB9ef/exnFBQU8N577/Huu+/yla98haqqKn7+85/zwgsvsGLFCqZNm5Z0WS644AJ+/etf8+677zJhwgR++tOfcsIJJ8TPjpK1m7Rz506mTZvGBx98wJFHHslPf/pTwGku4dZbb2X58uXcdNNNXHnllUnrLRwO89Zbb3HzzTfHp73jjjsoKipi1apV/OxnP2P58uVJp73qqqsoLy/n/fffp7m5maeffhqAiy66iFtvvZV33nlnl2nef/99HnvsMcrLy/nxj39MTk4Ob7/9NrNmzdrlktV3vvMdhg0bxtKlS1m6dCm/+tWv4g0SPvDAA52u8y1btnD99dfz2muv8eqrr3b5Vro9VXc3fD90/1ubDmbApakV6l2cccYZ+P3+Lsd58cUXU2p+eejQofFxBsUCxmnAraCgAIBx48axYcMGRowYkbT544kTJwLJm1V+4YUXeOihh+LzLSoq4umnn2bVqlXxNnJCodAubSDV1dVRW1sbb/xtwYIFnHHGGd3Wjc/ni7djdP7553PaaafR2NjI66+/3m761tbWpNMnLkPszOrVV1/lu991mjUbP358fHk7Wrp0Kb/5zW9oampix44dlJWVccQRR1BbW8vs2c6bc7/+9a/HG4kDOOqoo8jPzyc/P5+CggK++lWnibQJEya0S7Kp6Gyd/+tf/2LOnDmUuI0fnnXWWXzyySc9mvdA6zIRqOrf3P/39U84xnQuTa1Q7yKx+elAIEA0Go1/jjVHvLvNLydrGrq75o9TbVZZVZk7d+5ut46ZChEhGo1SWFiYUps9vW0auqWlhSuvvJJly5YxYsQIFi5c2KOmocFJYolNRfemaehk6/yJJ57o0Xz2RN1d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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Based on the trained models we can compare the test accuracy for each form of binary classification. Logistic Regression concluded with 94.377% accuracy on predicting the test data, K-Nearest Neighbors ended with 93.125%, and the Decision Tree with 90.625%. Though Decision Trees tend to have better accuracy by splitting features, they can also be prone to overfitting, especially with data that has a large number of features and less entries. In our case the features sport a linear trend, higher scores were indicative of higher chance of getting in, so it makes sense that Linear Regression sported the best results."
      ],
      "metadata": {
        "id": "Z8OTvezSXPCm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Additional Considerations for Machine Learning in the University Admissions Process\n",
        "Another example of using machine learning in the admissions process is predicting how likely different students are to enroll after being admitted. Universities must determine how many students to admit by factoring in an expected number of students who will decline admissions. Previous works have focused on developing models to predict admitted students' commitment decisions. To learn more about this approach, visit https://doi.org/10.3390/data4020065.\n",
        "\n",
        "An important factor to consider when evaluating models applied to this topic is fairness in machine learning. Diversity and bias in university admissions has been a hot topic for many years now. University admissions is largely a human decision process, with an admissions officer at each university reading applications and ultimately deciding yes or no. Any model that will predict admissions will be trained on historical data that relies on the biases of admissions officers. A model will learn these biases and trends. If a biased model is used on the admissions office end to make decisions, it will reinforce these biases in who will be admitted. If a biased model is used on the applicant end to determine which schools to apply to, this may eforce a cycle that recommends certain groups of people apply to and therefor attend ceratin categories of schools, also reinforceing the biases in the system. This is an ongoing challenge in machine learning that many in the field are working to address. Read about two current projects at the University of Maryland aimed at improving fairness in AI used for admissions and language translation here: https://ischool.umd.edu/news/improving-fairness-and-trust-in-ai-used-for-college-admissions-and-language-translation/.\n"
      ],
      "metadata": {
        "id": "NuYTaeB1butL"
      }
    }
  ]
}