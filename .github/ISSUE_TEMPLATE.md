* xclim version:
* Python version:
* Operating System:

### Description
<!--Describe what you were trying to get done.
Tell us what happened, what went wrong, and what you expected to happen.-->


### What I Did
<!--Paste the command(s) you ran and the output.
If there was a crash, please include the traceback below.-->
```
$ pip install foo --bar
```

### What I Received
<!--Paste the output or the stack trace of the problem you experienced here.-->
```
Traceback (most recent call last):
  File "/path/to/file/script.py", line 3326, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-9e1622b385b6>", line 1, in <module>
    1/0
ZeroDivisionError: division by zero
```
