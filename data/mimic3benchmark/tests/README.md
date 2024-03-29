# Tests for checking whether the benchmark datasets were generated correctly

### Root directory
```bash
python -um mimic3benchmark.tests.hash_tables -d data/root -o root-final.pkl;
diff root-final.pkl mimic3benchmark/tests/resources/root-final.pkl;
```

### In-hospital mortality
```bash
python -um mimic3benchmark.tests.hash_tables -d data/in-hospital-mortality -o ihm.pkl;
diff ihm.pkl mimic3benchmark/tests/resources/ihm.pkl;
```

### Decompensation
```bash
python -um mimic3benchmark.tests.hash_tables -d data/decompensation -o decomp.pkl;
diff decomp.pkl mimic3benchmark/tests/resources/decomp.pkl;
```

### Length-of-stay
```bash
python -um mimic3benchmark.tests.hash_tables -d data/length-of-stay -o los.pkl;
diff los.pkl mimic3benchmark/tests/resources/los.pkl;
```

### Phenotyping
```bash
python -um mimic3benchmark.tests.hash_tables -d data/phenotyping -o pheno.pkl;
diff pheno.pkl mimic3benchmark/tests/resources/pheno.pkl;
```

### Multitasking
```bash
python -um mimic3benchmark.tests.hash_tables -d data/multitask -o multitasking.pkl;
diff multitasking.pkl mimic3benchmark/tests/resources/multitasking.pkl;
```