library(mvtnorm)
library('ggplot2')

means = list(c(-.5,-.5),c(-.5,.5),c(.5,-.5),c(.5,.5))
sigma = matrix(c(.025,0,0,.025), ncol=2)

vals = do.call('rbind', lapply(1:length(means), function(i){
	cluster_means = means[i]
	cluster_vals = data.frame(as.matrix(rmvnorm(100, mean=means[[i]], sigma)))
	names(cluster_vals) = c('x','y')	
	cluster_vals$cluster = i
	return(cluster_vals) 	
}))

ggplot(vals) + geom_point(aes(x=x,y=y, color=factor(cluster))) + theme_classic() + theme(legend.position="none")
