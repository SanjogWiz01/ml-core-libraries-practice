best=float("inf"); bad=0
for e,l in enumerate([.9,.7,.6,.61,.63,.65],1):
 if l<best: best=l; bad=0
 else: bad+=1
 print(e,l)
 if bad>=3: print("Early stopping"); break