import operator
import sys
from collections import Counter
from pathlib import Path
from time import time


MAX_JOLT_DIFF = 3
MAX_CONNECTIONS = sys.maxsize
jolt_differences = Counter()

start = time()

connections_per_node = Counter()


def solve2(sample):
    node_degree = Counter()
    fname = 'sample.txt' if sample else 'input.txt'
    file = Path(f'puzzle_input/day10/{fname}')
    small_ex = """16
10
15
5
1
11
7
19
6
12
4"""
    numbers = map(int, file.read_text().splitlines())
    # numbers = map(int, small_ex.splitlines())
    jolts = sorted(list(numbers))
    jolts.insert(0, 0)
    jolts.append(jolts[-1] + 3)

    # determine the out-degree of each vertex
    for i in range(len(jolts)):
        curr_jolt = jolts[i]
        max_adapter = curr_jolt + MAX_JOLT_DIFF

        for j in range(i + 1, len(jolts)):
            adapter_candidate = jolts[j]
            jolt_diff = adapter_candidate - curr_jolt
            if adapter_candidate > max_adapter:
                break
            elif jolt_diff != 0:
                connections_per_node[curr_jolt] += 1
                print(f'{adapter_candidate}(+{jolt_diff}),', end='')
                jolt_differences[adapter_candidate - curr_jolt] += 1
                node_degree[curr_jolt] += 1
    print()
    print("Node Degrees:")
    node_degree_pairs = sorted(node_degree.items(), key=operator.itemgetter(0))

    clusters = form_clusters(node_degree_pairs)
    connections_total = 1

    for cluster in clusters:
        cluster_count = explore_jolts(cluster)
        connections_total *= cluster_count

    print(connections_total)

    return connections_total


def form_clusters(node_degrees):
    """Build clusters of nodes that have out-degree > 1
    End the cluster on next highest node with out-degree 1 where the subsequent node also has out-degree 1
    """
    clusters = []
    cluster = []
    for i, pair in enumerate(node_degrees):

        # outdegree of this node and next are both one, next node ends this cluster,
        # alternatively, this node ends cluster if we're at list end
        if pair[1] == 1 and cluster:
            cluster.append(pair[0])

            try:
                assert node_degrees[i + 1][1] == 1
            except (KeyError, AssertionError):
                pass
            else:
                cluster.append(node_degrees[i+1][0])

            clusters.append(cluster)
            cluster = []
        # if we're already building a cluster, keep building
        # or start a new one if this node's out-degree is higher than 1
        elif cluster or pair[1] > 1:
            cluster.append(pair[0])

    return clusters


def explore_jolts(jolts):
    """Return the number of unique max 3 jump paths between jolts[0] and jolts[-1]"""
    return _explore_adapters(jolts[0], jolts, list(), jolts[-1])


def _explore_adapters(curr_jolt, jolts, visited_jolts, final_jolt):
    arrangements = 0

    visited_jolts.append(curr_jolt)

    for j in range(len(jolts)):
        jolt_candidate = jolts[j]
        if jolt_candidate > curr_jolt + MAX_JOLT_DIFF:
            break
        elif jolt_candidate - curr_jolt != 0 and jolt_candidate not in visited_jolts:
            arrangements += _explore_adapters(jolt_candidate, jolts[j + 1:], visited_jolts, final_jolt)

    if curr_jolt == final_jolt:
        arrangements += 1

    if visited_jolts:
        visited_jolts.pop()

    return arrangements


if __name__ == '__main__':
    assert solve2(False) == 32396521357312
