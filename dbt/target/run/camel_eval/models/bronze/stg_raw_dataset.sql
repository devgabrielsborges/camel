
  
  create view "camel"."main"."stg_raw_dataset__dbt_tmp" as (
    select * from read_parquet('../data/raw/train.parquet')
  );
