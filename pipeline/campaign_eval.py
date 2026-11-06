import json,pathlib,sys
import campaign_policies as p
import benchmark_agents as b
import statistics,math

def ci_stats(values):
 from scipy.stats import t
 n=len(values);mean=statistics.fmean(values);se=statistics.stdev(values)/math.sqrt(n) if n>1 else 0.
 critical=float(t.ppf(.975,n-1)) if n>1 else 0.
 return {'mean':mean,'se':se,'ci95_low':mean-critical*se,'ci95_high':mean+critical*se,'n':n}

def main():
 job=json.loads(pathlib.Path(sys.argv[1]).read_text());p.install()
 b.FAMILIES=tuple(job.pop('families'));b.sample_spec=p.sample_spec;b.choose=p.choose;b.ci_stats=ci_stats
 dest=pathlib.Path(job.pop('result_path'))
 result=b.candidate_job(job);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(dest)

if __name__=='__main__':main()
