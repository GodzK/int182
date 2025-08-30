
data <- c(34, 15, 20, 2, -14, 19, 25, 40, 33, 17, 9, 60, 19, 25, 62)
sorted_data <- sort(data)
print("Sorted data:")
print(sorted_data)
summary(data)
Q1 <- quantile(data, 0.25)
Q2 <- median(data)
Q3 <- quantile(data, 0.75)
IQR_val <- IQR(data)
lower_fence <- Q1 - 1.5 * IQR_val
upper_fence <- Q3 + 1.5 * IQR_val
cat("Q1 =", Q1, "\n")
cat("Q2 =", Q2, "\n")
cat("Q3 =", Q3, "\n")
cat("IQR =", IQR_val, "\n")
cat("Lower Fence =", lower_fence, "\n")
cat("Upper Fence =", upper_fence, "\n")
outliers <- data[data < lower_fence | data > upper_fence]
cat("Outliers =", outliers, "\n")
boxplot(data, horizontal = TRUE, main = "Boxplot of data")
hist(data, main = "Histogram of data", xlab = "data")
