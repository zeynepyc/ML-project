# ML-project

## Introduction
For our final project, we will attempt to predict chances of university admissions. The data we will be using is the "Data for Admission in the University" dataset from Kaggle. You can check it out from this link: https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university.

## Motivation
Our main motivation is the financial burden of college applications on students. College Board recommends students to apply to 5 to 8 colleges. On average, the cost of one application is around 50$ and can go as high as 80$. This can add up to a total cost of 250$ - 640$ which is a lot to handle for a high school student. As we all know, college admissions can be tricky and stressful. Although students can roughly have an idea about their chances of being admitted to a particular university, decisions are often unexpected. Having a more accurate way to predict chances of admissions can help students make better decisions while deciding which schools to apply to, thus maximize their chances with the cost they pay.

## Binary Classification
Binary classification is one of the most popular implementations of machine learning. It is a supervised learning algorithm which classifies a set of examples into two groups or classes. The prediction is based on a the chosen binary classification algorithm. We will explore several of these algorithms such as:
<br>
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier
Our problem can easily be cast as a binary classification task.
<br>
Our dataset includes a 'Chance of Admit' column which represents the probability that a student will be granted admission to the university. We can map these percentages to another column which marks a row as 0 if the percentage is less than 0.5, and 1 otherwise, where 0 represents that the student is unlikely to be admitted and 1 is the contrary.

