#!/usr/bin/env python3
from round029_online_core import R,collect,write,read

def main():
 assert not (R/'experiment-at-report.json').exists();s=collect();write(R/'experiment-at-report.json',s)
 arm=s['status'].get('current_job');log=R/((arm or 'rl43')+'-train.log');(R/'training-log-at-report.txt').write_text(log.read_text(errors='replace') if log.exists() else '')
 print({k:s[k] for k in ['captured_at_utc','independently_rescored','logged_native_steps','actual_completed_learning_steps','status']})
if __name__=='__main__':main()
