# DeepMutations
CABBI project



####TODO list

-[ ] Multiple models, top10,20,30 AND 100 -> Majority voting (ones that models agree)
-[ ] Rank by score for all models, either keep top-1000 and find the top-100 common ones in the list 
or keep the ones that are positive for each model (based on the threshold) and then pick the top100 common ones
-[x] Use all training data when testing the unseen 
-[ ] Brief explanation of the process
-[ ] Cross validation for ensemble, test on top10-top30-top100-top200
-[ ] USE ONLY RF + PRINT SUPPORTING SCORES (FOR OLD LIST)


-[x] Train top-10, evaluate with top-30
-[ ] Train with very clearly high score instances: take out ambiguous ones (in the middle)
-[ ] Avoid ties in LamdaMart, how to constraint to reduce ties
-[x] Print ids and send them to Michael
-[x] Make a regression model (as before) but evaluate on ranking scenario 


-[ ] fij our target, {fi, fj} features
-[ ] Text data from the ontology database: check with text data solely, with  {fi, fj} solely, or with both features sets
-[ ] LambdaMart as NN -- Almost done

-[x] stricter criteria count only top 100 as good ones (1 for only top 10/30)
-[x] print items
-[x] check features
-[x] Target values beyond 1.0
-[x] NDCG for training but also report Precision@10
-[x] Evaluation on multiple cutoff points: top10, top20 
-[ ] Optional: IRGAN https://github.com/hwang1996/IRGAN-pytorch http://lantaoyu.com/files/2017-04-19-gans-for-ir.pdf
-[ ] Optional: RankSVM, polynomial features
