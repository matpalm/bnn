library(ggplot2)
library(lubridate)

df = read.delim("/home/mat/dev/bnn2/predict.out", h=F,
                col.names=c("idx", "img_name", "count"))
df$idx = NULL
head(df)

df$dts = ymd_hms(gsub(".jpg", "", df$img_name))
df$date_str = as.factor(sprintf("%4d%02d%02d", year(df$dts), month(df$dts), day(df$dts)))
df$time_of_day = hour(df$dts) + minute(df$dts)/60 + second(df$dts)/3600
summary(df)

ggplot(df, aes(time_of_day, count)) + 
  geom_point(alpha=0.3, aes(color=date_str), position = "jitter") + 
  geom_smooth(aes(color=date_str)) +
  facet_grid(.~date_str)
