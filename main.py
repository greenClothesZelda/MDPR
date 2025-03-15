import IO
import llm


def main(question):
    Q = question
    main_reference, sub_reference = .get_main_references(Q)
    response = llm.main(Q,main_reference, sub_reference)

    Q_res_list = []
    Q_res_list.append(Q, response)
if __name__ == "__main__":
    main()