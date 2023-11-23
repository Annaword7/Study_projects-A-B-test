# Задание ENGLISH VERSION DESCRPTN BELOW

---

Представьте, что вы работаете аналитиком в очень крупной компании по доставке пиццы над приложением для курьеров (да, обычно в таких компаниях есть приложение для курьеров и отдельно приложение для потребителей).

У вас есть несколько ресторанов в разных частях города и целый штат курьеров. Но есть одна проблема — к вечеру скорость доставки падает из-за того, что курьеры уходят домой после рабочего дня, а количество заказов лишь растет. Это приводит к тому, что в момент пересмены наша доставка очень сильно проседает в эффективности.

Наши data scientist-ы придумали новый алгоритм, который позволяет курьерам запланировать свои последние заказы перед окончанием рабочего дня так, чтобы их маршрут доставки совпадал с маршрутом до дома. То есть, чтобы курьеры доставляли последние свои заказы за день как бы "по пути" домой.

Вы вместе с командой решили раскатить A/B тест на две равные группы курьеров. Часть курьеров использует старый алгоритм без опции "по пути", другие видят в своем приложении эту опцию и могут ее выбрать. Ваша задача – проанализировать данные эксперимента и помочь бизнесу принять решение о раскатке новой фичи на всех курьеров.

<aside>
💡 Описание данных
order_id - id заказа
delivery_time - время доставки в минутах
district - район доставки
experiment_group - экспериментальная группа

</aside>

# Как действуем?

Для сравнения будут использованы данные о времени доставки в контрольной и тестововй группе

1. Формулируем гипотезы:
- Нулевая гипотеза (H0): Разницы между средним временем доставки в тестовой и контрольной группе нет
- Альтернативная гипотеза (H1): Разница между средним временем доставки в тестовой и контрольной группе есть
1. Критерий, который позволяет сравнивать две выборки между собой (два выборочных средних), называется t-тест (t-test), или просто t-критерий Стьюдента.
Особенно важный вопрос — это требование к нормальности данных обеих групп при применении t-теста.
Во многих случаях можно встретить довольно жесткое требование к нормальности данных по причине возможного завышения вероятности ошибки I рода.

---

<aside>
💡 NB! На практике t-тест может быть использован для сравнения средних и при ненормальном распределении, особенно на больших выборках и если в данных нет заметных выбросов.
Однако при этом вы выходите на очень тонкий лёд — перед использованием t-теста на ненормальных данных дважды подумайте о своих жизненных решениях.
Возможно, непараметрический тест или бутстрап окажутся лучше и адекватнеею
Как вариант, можно преобразовать переменную, например, логарифмировать, чтобы сделать распределение более симметричным.

</aside>

---

# Task

---

Imagine that you are working as an analyst for a very large pizza delivery company on an application for couriers (yes, usually such companies have an application for couriers and a separate application for consumers).

You have several restaurants in different parts of the city and a whole staff of couriers. But there is one problem — by the evening the delivery speed drops due to the fact that couriers go home after a working day, and the number of orders is only growing. This leads to the fact that at the time of the shift, our delivery is very much sagging in efficiency.

Our data scientists have come up with a new algorithm that allows couriers to schedule their last orders before the end of the working day so that their delivery route coincides with the route home. That is, so that couriers deliver their last orders for the day as if "on the way" home.

You and your team decided to roll out the A/B test into two equal groups of couriers. Some couriers use the old algorithm without the option "on the way", others see this option in their application and can choose it. Your task is to analyze the experiment data and help the business make a decision about rolling out a new feature to all couriers.

<aside>
💡 Data description
order_id - order
id delivery_time - delivery time in minutes
district - delivery
area experiment_group - experimental group

</aside>

# How do we act?

For comparison, data on the delivery time in the control and test groups will be used

1. We formulate hypotheses:
- Null hypothesis (H0): There is no difference between the average delivery time in the test and control group
- Alternative hypothesis (H1): There is a difference between the average delivery time in the test and control group
1. The criterion that allows you to compare two samples with each other (two sample averages) is called the t-test, or simply the Student's t-test.
A particularly important issue is the requirement for the normality of the data of both groups when applying the t-test.
In many cases, it is possible to meet a rather strict requirement for the normality of data due to the possible overestimation of the probability of an error of the first kind.

---

<aside>
💡 NB! In practice, the t-test can be used to compare averages and with an abnormal distribution, especially on large samples and if there are no noticeable outliers in the data.
However, at the same time you are on very thin ice — before using the t-test for abnormal data, think twice about your life decisions.
Perhaps a nonparametric test or bootstrap will be better and more adequate
Alternatively, you can transform the variable, for example, logarithm it to make the distribution more symmetrical.

</aside>

---

From the T-test, we want to get a p-significance level greater than 0.05, because if it is less, then the news is bad:
we are testing the hypothesis that the distributions do not significantly differ from normal, so we do not want to reject H_0, which means that the new model has not become more effective.

Let's calculate in practice how much the new model is better or worse than the previous one.

Из Т-теста мы хотим получить р-уровень значимости больше 0.05, поскольку если он меньше, то новость плохая:
мы тестируем гипотезу о том, что распределения значимо не отличаются от нормального, поэтому отклонять H_0 мы не хотим, а значит новая модель не стала более эффективной.

Посчитаем же на практике, насколько новая модель лучше или хуже прежней.
