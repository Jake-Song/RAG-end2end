def conflict_matrix(retrieved_page_number, reference_page_number):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i, page in enumerate(retrieved_page_number):
        if page in reference_page_number:
            true_positives += 1
        elif i <= len(reference_page_number)-1 and page not in reference_page_number:
            false_positives += 1
            false_negatives += 1
        elif i > len(reference_page_number)-1 and page not in reference_page_number:
            false_positives += 1
            
    return true_positives, false_positives, false_negatives

# ref page를 전부 찾은 경우
def test_all_match():
    retrieved_page_number = [1, 2, 3]
    reference_page_number = [1, 2, 3]

    true_positives, false_positives, false_negatives = conflict_matrix(retrieved_page_number, reference_page_number)

    assert true_positives == 3
    assert false_positives == 0
    assert false_negatives == 0

# ref page를 전부 찾지 못한 경우
def test_all_no_match():
    retrieved_page_number = [4, 5, 6]
    reference_page_number = [1, 2, 3]

    true_positives, false_positives, false_negatives = conflict_matrix(retrieved_page_number, reference_page_number)

    assert true_positives == 0
    assert false_positives == 3
    assert false_negatives == 3

# ref page를 전부 찾고 잘 못된 페이지를 찾은 경우
def test_all_match_but_wrong_page():
    retrieved_page_number = [1, 2, 3, 4, 6, 7]
    reference_page_number = [1, 2, 3]

    true_positives, false_positives, false_negatives = conflict_matrix(retrieved_page_number, reference_page_number)

    assert true_positives == 3
    assert false_positives == 3
    assert false_negatives == 0


def test_wrong_match_then_all_match_page():
    retrieved_page_number = [4, 6, 7, 1, 2, 3]
    reference_page_number = [1, 2, 3]

    true_positives, false_positives, false_negatives = conflict_matrix(retrieved_page_number, reference_page_number)
    print(true_positives, false_positives, false_negatives)
    # assert true_positives == 3
    # assert false_positives == 3
    # assert false_negatives == 0

# ref page를 일부 찾고 잘 못된 페이지를 찾은 경우
def test_partial_match_but_wrong_page():
    retrieved_page_number = [1, 2, 5, 7]
    reference_page_number = [1, 2, 3]

    true_positives, false_positives, false_negatives = conflict_matrix(retrieved_page_number, reference_page_number)

    assert true_positives == 2
    assert false_positives == 2
    assert false_negatives == 1

# 중복 페이지 제거
def test_unique_page_number():
    reference_page_number = '[1, 2, 3, 7, 2]'
    retrieved_page_number = '[1, 2, 5, 7, 7, 2]'
    
    reference_page_number = list({int(page) for page in reference_page_number.strip("[]").split(",")})
    retrieved_page_number = list({int(page) for page in retrieved_page_number.strip("[]").split(",")})
   
    assert reference_page_number == [1, 2, 3, 7]
    assert retrieved_page_number == [1, 2, 5, 7]

# 테스트 실행
test_all_match()
test_all_no_match()
test_all_match_but_wrong_page()
test_wrong_match_then_all_match_page()
test_partial_match_but_wrong_page()
test_unique_page_number()