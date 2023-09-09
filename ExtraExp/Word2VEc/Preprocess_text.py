import pickle

def filter_text_file(input_file_path, output_file_path, word_to_id):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    filtered_lines = []

    for line in lines:
        words = line.strip().split()
        valid_line = all(word in word_to_id for word in words)
        if valid_line:
            filtered_lines.append(line)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(filtered_lines)

# 예시 사용법
with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
input_file_path = 'C:\Users\Suseong Kim\Desktop\MILAB_VENV\ExtraExp\Word2VEc\questions-words.txt'  # 입력 파일 경로를 넣어주세요.

output_file_path = 'C:\Users\Suseong Kim\Desktop\MILAB_VENV\ExtraExp\Word2VEc\preprocessed_QW.txt'  # 출력 파일 경로를 넣어주세요.

filter_text_file(input_file_path, output_file_path, word_to_id)
