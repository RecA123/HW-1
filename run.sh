set -e

cd "$(dirname "$0")"
PYTHON=python

check_nyt() {
    if [ ! -f "nyt.csv" ]; then
        echo "Error: nyt.csv not found."
        exit 1
    fi
}
check_task2_data() {
    check_nyt

    if [ ! -f "ag.csv" ]; then
        echo "Error: ag.csv not found."
        exit 1
    fi

    if [ ! -f "glove.6B/glove.6B.100d.txt" ]; then
        echo "Error: GloVe 100d file not found."
        exit 1
    fi
}

run_task1() {
    echo "=================================="
    echo "Running Task 1: Bag of Words"
    echo "=================================="

    check_nyt
    "$PYTHON" T1.py

    echo "Task 1 completed."
}

run_task2() {
    echo "=================================="
    echo "Running Task 2: Word Embeddings"
    echo "=================================="

    check_task2_data
    "$PYTHON" T2.py

    echo "Task 2 completed."
}

run_task3() {
    echo "=================================="
    echo "Running Task 3: BERT"
    echo "=================================="

    check_nyt
    "$PYTHON" T3.py

    echo "Task 3 completed."
}


case "${1:-all}" in
    1)
        run_task1
        ;;
    2)
        run_task2
        ;;
    3)
        run_task3
        ;;
    all)
        run_task1
        run_task2
        run_task3
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Error: Invalid argument."
        show_help
        exit 1
        ;;
esac

echo "=================================="
echo "Finished!"
echo "=================================="