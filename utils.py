import re

def preprocess_text(text: str, stop_words: list[str]=[]):

	# Remove leading and trailing white spaces
	text = text.strip()

	# Convert non-ASCII quotes to ASCII quotes
	text = re.sub(r"‘|’", "'", text)
	text = re.sub(r"“|”", '"', text)

	# Remove non-ASCII characters
	text = re.sub(r"[^\x00-\x7f]+", "", text)

	# Remove emails
	text = re.sub(r"[^\s]+@[^\s]+\.com", "", text)

	# Remove hyperlinks
	text = re.sub(r"[^\s]*://[^\s]*", "", text)

	# Remove hashtags
	text = re.sub(r"#[^\s]+", "", text)

	# Remove HTML tags
	text = re.sub(r"<[^\n>]+>", "", text)

	# Remove numbers
	# text = re.sub(r"[+?\d+-?]+", "", text)

	# Remove stop words
	if stop_words:
		text = re.sub(r"|".join([
			rf"\s{word}([\s.!:;])" for word in stop_words
		]), r"\1", text)

	# Concatenating multiple spaces and tabs
	text = re.sub(r"[ \t]{2,}", " ", text)

	# Removing spaces and tabs before newline
	text = re.sub(r"[ \t]\n", "\n", text)

	# Concatenating multiple newlines
	text = re.sub(r"\n{3,}", "\n\n", text)

	return text
