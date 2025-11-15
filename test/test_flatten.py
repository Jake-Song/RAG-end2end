# Test Code for ID and Page Number Merging Logic
import copy

def test_id_page_merging():
    """Test the logic that merges element IDs and page numbers across multiple JSON data arrays"""
    
    # Test Case 1: Basic test with two data chunks
    print("=" * 60)
    print("Test Case 1: Basic merging of two data chunks")
    print("=" * 60)
    
    json_data_arr = [
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'Element 1'},
                {'id': 1, 'page': 1, 'text': 'Element 2'},
                {'id': 2, 'page': 2, 'text': 'Element 3'}
            ]
        },
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'Element 4'},
                {'id': 1, 'page': 2, 'text': 'Element 5'}
            ]
        }
    ]
    
    # Make a copy to preserve original
    test_data = copy.deepcopy(json_data_arr)
    
    # Run the logic
    last_id, last_page = None, None
    
    for data in test_data:
        for idx, element in enumerate(data['elements']):
            if last_id is not None and last_page is not None:
                start_id = last_id + 1
                element['id'] = start_id + element['id']
                element['page'] = last_page + element['page']
            
            if idx == len(data['elements']) - 1:
                last_id = element['id']
                last_page = element['page']
    
    # Verify results
    print("Results:")
    for i, data in enumerate(test_data):
        print(f"\nData chunk {i}:")
        for element in data['elements']:
            print(f"  ID: {element['id']}, Page: {element['page']}, Text: {element['text']}")
    
    # Expected IDs: 0, 1, 2, 3, 4
    # Expected Pages: 1, 1, 2, 3, 4
    expected_ids = [0, 1, 2, 3, 4]
    expected_pages = [1, 1, 2, 3, 4]
    
    actual_ids = []
    actual_pages = []
    for data in test_data:
        for element in data['elements']:
            actual_ids.append(element['id'])
            actual_pages.append(element['page'])
    
    assert actual_ids == expected_ids, f"ID mismatch! Expected {expected_ids}, got {actual_ids}"
    assert actual_pages == expected_pages, f"Page mismatch! Expected {expected_pages}, got {actual_pages}"
    print("\n✓ Test Case 1 PASSED")
    
    # Test Case 2: Three data chunks with varying sizes
    print("\n" + "=" * 60)
    print("Test Case 2: Three data chunks with varying sizes")
    print("=" * 60)
    
    json_data_arr = [
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'A1'},
                {'id': 1, 'page': 1, 'text': 'A2'}
            ]
        },
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'B1'},
                {'id': 1, 'page': 1, 'text': 'B2'},
                {'id': 2, 'page': 2, 'text': 'B3'}
            ]
        },
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'C1'}
            ]
        }
    ]
    
    test_data = copy.deepcopy(json_data_arr)
    
    # Run the logic
    last_id, last_page = None, None
    
    for data in test_data:
        for idx, element in enumerate(data['elements']):
            if last_id is not None and last_page is not None:
                start_id = last_id + 1
                element['id'] = start_id + element['id']
                element['page'] = last_page + element['page']
            
            if idx == len(data['elements']) - 1:
                last_id = element['id']
                last_page = element['page']
    
    print("Results:")
    for i, data in enumerate(test_data):
        print(f"\nData chunk {i}:")
        for element in data['elements']:
            print(f"  ID: {element['id']}, Page: {element['page']}, Text: {element['text']}")
    
    # Expected IDs: 0, 1, 2, 3, 4, 5
    # Expected Pages: 1, 1, 2, 2, 3, 4
    expected_ids = [0, 1, 2, 3, 4, 5]
    expected_pages = [1, 1, 2, 2, 3, 4]
    
    actual_ids = []
    actual_pages = []
    for data in test_data:
        for element in data['elements']:
            actual_ids.append(element['id'])
            actual_pages.append(element['page'])
    
    assert actual_ids == expected_ids, f"ID mismatch! Expected {expected_ids}, got {actual_ids}"
    assert actual_pages == expected_pages, f"Page mismatch! Expected {expected_pages}, got {actual_pages}"
    print("\n✓ Test Case 2 PASSED")
    
    # Test Case 3: Single data chunk (edge case)
    print("\n" + "=" * 60)
    print("Test Case 3: Single data chunk (no merging needed)")
    print("=" * 60)
    
    json_data_arr = [
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'Only 1'},
                {'id': 1, 'page': 2, 'text': 'Only 2'}
            ]
        }
    ]
    
    test_data = copy.deepcopy(json_data_arr)
    
    # Run the logic
    last_id, last_page = None, None
    
    for data in test_data:
        for idx, element in enumerate(data['elements']):
            if last_id is not None and last_page is not None:
                start_id = last_id + 1
                element['id'] = start_id + element['id']
                element['page'] = last_page + element['page']
            
            if idx == len(data['elements']) - 1:
                last_id = element['id']
                last_page = element['page']
    
    print("Results:")
    for element in test_data[0]['elements']:
        print(f"  ID: {element['id']}, Page: {element['page']}, Text: {element['text']}")
    
    # IDs and pages should remain unchanged
    expected_ids = [0, 1]
    expected_pages = [1, 2]
    
    actual_ids = [e['id'] for e in test_data[0]['elements']]
    actual_pages = [e['page'] for e in test_data[0]['elements']]
    
    assert actual_ids == expected_ids, f"ID mismatch! Expected {expected_ids}, got {actual_ids}"
    assert actual_pages == expected_pages, f"Page mismatch! Expected {expected_pages}, got {actual_pages}"
    print("\n✓ Test Case 3 PASSED")
    
    # Test Case 4: Empty elements list (edge case)
    print("\n" + "=" * 60)
    print("Test Case 4: Data chunk with empty elements")
    print("=" * 60)
    
    json_data_arr = [
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'First'}
            ]
        },
        {
            'elements': []
        },
        {
            'elements': [
                {'id': 0, 'page': 1, 'text': 'Second'}
            ]
        }
    ]
    
    test_data = copy.deepcopy(json_data_arr)
    
    # Run the logic
    last_id, last_page = None, None
    
    for data in test_data:
        for idx, element in enumerate(data['elements']):
            if last_id is not None and last_page is not None:
                start_id = last_id + 1
                element['id'] = start_id + element['id']
                element['page'] = last_page + element['page']
            
            if idx == len(data['elements']) - 1:
                last_id = element['id']
                last_page = element['page']
    
    print("Results:")
    all_elements = []
    for i, data in enumerate(test_data):
        if data['elements']:
            print(f"\nData chunk {i}:")
            for element in data['elements']:
                print(f"  ID: {element['id']}, Page: {element['page']}, Text: {element['text']}")
                all_elements.append(element)
        else:
            print(f"\nData chunk {i}: (empty)")
    
    # Expected IDs: 0, 1
    # Expected Pages: 1, 2
    expected_ids = [0, 1]
    expected_pages = [1, 2]
    
    actual_ids = [e['id'] for e in all_elements]
    actual_pages = [e['page'] for e in all_elements]
    
    assert actual_ids == expected_ids, f"ID mismatch! Expected {expected_ids}, got {actual_ids}"
    assert actual_pages == expected_pages, f"Page mismatch! Expected {expected_pages}, got {actual_pages}"
    print("\n✓ Test Case 4 PASSED")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)

# Run the tests
test_id_page_merging()