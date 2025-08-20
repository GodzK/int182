# INT182
# SIT@KMUTT
# 2025


# Let's use the iris data already given in R
data("iris")

# To understand the data, Iris data set, visit:
# https://www.kaggle.com/uciml/iris

# import the graphics plot library
library(ggplot2)


# plot the info of sepal
myplotSepal <- ggplot(iris_data, aes(x=SepalLengthCm, y=SepalWidthCm, color=Species)) +
  geom_point(size=2) +
  theme_light(base_size=16) + 
  ggtitle("Sepal Width vs. Sepal Length")
myplotSepal

# plot the info of petal
myplotPetal <- ggplot(iris_data, aes(x=PetalWidthCm, y=PetalLengthCm, color=Species))+
  geom_point(size=2) +
  theme_light(base_size=16) + 
  ggtitle("Petal Length vs. Petal Width")
myplotPetal

# show different species in different shapes
anotherPlot <- ggplot(iris_data, aes(x=PetalLengthCm, y=PetalWidthCm, color=Species, shape=Species)) + 
  geom_point(size=3.5, alpha=0.4, show.legend = TRUE) +
  ggtitle("Petal length vs. petal width") +
  xlab("Petal length (cm)") +
  ylab("Petal width (cm)") +
  theme(
    plot.title=element_text( hjust=0.5, vjust=0.5, face='bold')
  )
anotherPlot

# save the plot
ggsave("Petal.png", anotherPlot, height=6, width=10, units="in")

# now try to see the box plot (box and whisker plot))
# boxplot
bxplotPetalW <- boxplot(PetalWidthCm ~ Species, data=iris_data, horizontal = FALSE, col = rgb(.2,.5,0,0.5))
bxplotPetalL <- 
  boxplot(PetalLengthCm ~ Species, data=iris_data, horizontal = TRUE, col = rgb(.8,.3,0,0.5))


# select just some species
iris_setosa <- iris_data[which(iris_data$Species == "Iris-setosa"),]

# try a simple histogram plot
hist(iris_setosa$SepalWidthCm, col = "tan")
# and a simple box plot
boxplot(iris_setosa$PetalLengthCm, col = "lightgreen")

# try to save, using a simple approach
hi <- hist(iris_setosa$SepalWidthCm, col = "blue")
png(filename ="hi.png")
plot(hi, col = "green")
dev.off()


