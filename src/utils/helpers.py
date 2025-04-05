"""Utility functions for the QA system."""
from functools import partial
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

def print_runnable(preface=""):
    """Create a runnable that prints then returns its input."""
    def print_and_return(x, preface):
        if preface: 
            print(preface, end="")
        print(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def save_memory_and_get_output(d, vstore):
    """Save input/output to conversation store and return output."""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')