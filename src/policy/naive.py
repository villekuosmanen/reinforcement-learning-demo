# always drive forward
def policy(env):
    # We can take one of three actions:
    # - drive backwards: 0
    # - do nothing: 1
    # - drive forward: 2
    # the naive policy just attempts to drive forward.
    return 2
