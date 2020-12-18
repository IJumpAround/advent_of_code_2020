from functools import reduce


def run(data=None):
    a_z = [chr(i) for i in range(97, 97 + 26)]

    if not data:
        f = open('puzzle_input/day6/input.txt', 'r')
        data = "".join([line for line in f])
        f.close()

    groups = data.split('\n\n')
    print(repr(data))
    print(groups)

    group_answers = []
    for group in groups:
        group = group.splitlines()
        questions = set()
        person_questions = []
        for person in group:
            person_question = set()
            for question in person:
                person_question.add(question)
            person_questions.append(person_question)
        group_answers.append(set(reduce(lambda x,y: x & y, person_questions)))

        group_answers.append(questions)


    ans_count = 0
    for group_ans in group_answers:
        ans_count += sum([1 for i in range(len(group_ans))])

    print(ans_count)


if __name__ == '__main__':
    run()