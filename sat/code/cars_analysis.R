rm(list = ls())
getwd()
data=read.csv("analysis.csv", header=TRUE)
colnames(data)
data$Interact=data$Manual.Count*as.numeric(data$State)
fit_base=lm(as.numeric(Total.Rev)~ Manual.Count*State+Azimuth,data=data)
AIC(fit_base)
summary(fit_base)
anova(fit_base)
plot(fit_base)

library(car)
fit_int=lm(as.numeric(Total.Rev)~ Manual.Count+State+Interact+Azimuth,data=data)

crPlots(fit_int)
# Ceres plots 
ceresPlots(fit_int)

fit_w=lm(as.numeric(Total.Rev)~ Manual.Count*State+Azimuth,data=data,weights=1/(fit_base$resi)**2)

summary(fit_w)
anova(fit_w)

plot(fit_w)


library(MASS)
data=read.csv("analysis.csv", header=TRUE)

data$Total.Rev=as.numeric(data$Total.Rev)
fit=lm(as.numeric(Total.Rev)~ Manual.Count*Azimuth,data=data)
a = boxcox(fit)
b=a$x[which.max(a$y)] ###best transformation parameter
data$Ynew = data$Total.Rev^{b}
fit.new = lm(Ynew ~ Manual.Count*State+Azimuth, data=data)
AIC(fit.new)
coef(fit.new)
summary(fit.new)
anova(fit.new)
plot(fit.new)
library(car)
vif(fit.new)

#install.packages("pwr")
library(pwr)

pwr.f2.test(16,23,1)
pwr.anova.test(5,40,.5)
pwr.t.test(n=36,d=1)
pwr.t.test(n=36,d=.0003)
