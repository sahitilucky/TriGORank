print ("Hello World!!")
con <- file("~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small.txt", "r")
w<-readLines(con)
close(con)
library(GOSemSim)
ScGO_bp <- godata('org.Sc.sgd.db', ont="BP")
ScGO_mf <- godata('org.Sc.sgd.db', ont="MF")
ScGO_cc <- godata('org.Sc.sgd.db', ont="CC")
ScGOs <- c(ScGO_mf, ScGO_bp, ScGO_cc)
num_pairs <- length(w)
num_pairs <- length(w)
#cat('', file = "~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small_sims.txt", sep = "\t", fill = FALSE, labels = NULL,append = FALSE)
for (i in 10001:num_pairs){
	x <- strsplit(w[i], "\t")
	go1 <- x[[1]][1]
	go2 <- x[[1]][2]
	sim_list <- c()
	for (j in 1:3){
		wang_sim = goSim(go1, go2, semData = ScGOs[[j]], measure = "Wang")
		jiang_sim = goSim(go1, go2, semData = ScGOs[[j]], measure = "Jiang")
		lin_sim = goSim(go1, go2, semData = ScGOs[[j]], measure = "Lin")
		resnik_sim = goSim(go1, go2, semData = ScGOs[[j]], measure = "Resnik")
		rel_sim = goSim(go1, go2, semData = ScGOs[[j]], measure = "Rel")
		sim_list <- c(sim_list, wang_sim, jiang_sim, lin_sim, resnik_sim, rel_sim)
	}
	#write similarities to a file
	sim_list <- c(go1,go2,sim_list)
	#ncols <- length(sim_list)
	#lapply(sim_list, write, "~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small_sims.txt", append = TRUE, ncolumns = ncols, nrows = 1)
	#write.table(sim_list, file = "~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small_sims.txt", append= TRUE, sep = "\t", dec = ".",row.names = FALSE, col.names = FALSE)
	cat(sim_list, file = "~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small_sims.txt", sep = "\t", fill = FALSE, labels = NULL,append = TRUE)
	cat("\n", file = "~/Documents/DeepMutations/gene_pairs_data/go_term_pairs_small_sims.txt", fill = FALSE, append = TRUE)
	print (i)
}
		
		
	

	
	
	 
