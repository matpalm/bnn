library(ggplot2)
library(lubridate)

df1 = read.delim("/home/mat/dev/bnn2/20180204.out", h=F,
                 col.names=c("idx", "img_name", "l_min", "l_max", "count"))
df2 = read.delim("/home/mat/dev/bnn2/20180205.out", h=F,
                 col.names=c("idx", "img_name", "l_min", "l_max", "count"))
df3 = read.delim("/home/mat/dev/bnn2/20180206.out", h=F,
                 col.names=c("idx", "img_name", "l_min", "l_max", "count"))
df = rbind(df1, df2, df3)
df$l_min = NULL
df$l_max = NULL
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
