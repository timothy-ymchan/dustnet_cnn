python s3-long-multiproc-run.py  --in-path ../athenaMLdata/data/TurbPar.out2.00103.athdf --out-prefix ./s3long-out/s3-normal-multiproc-bash --r-bins "1/128,2/10,50" --nproc 5 --nsamples 1000000
python s3-long-multiproc-run.py  --in-path ../athenaMLdata/data_dedt5e-4/TurbPar.out2.00010.athdf --out-prefix ./s3long-out/s3-dedt5e4-multiproc-bash --r-bins "1/128,2/10,50" --nproc 5 --nsamples 1000000
python s3-long-multiproc-run.py  --in-path ../athenaMLdata/data_kdrive3-8/TurbPar.out2.00010.athdf --out-prefix ./s3long-out/s3-kdrive38-multiproc-bash --r-bins "1/128,2/10,50" --nproc 5 --nsamples 1000000
python s3-long-multiproc-run.py  --in-path ../athenaMLdata/data_ts0.001/TurbPar.out2.00010.athdf --out-prefix ./s3long-out/s3-ts001-multiproc-bash --r-bins "1/128,2/10,50" --nproc 5 --nsamples 1000000
