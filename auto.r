install.packages("ISLR")
library(ISLR)

# ใช้ dataset Auto
data(Auto)

# Linear regression
model <- lm(mpg ~ horsepower, data = Auto)
summary(model)
