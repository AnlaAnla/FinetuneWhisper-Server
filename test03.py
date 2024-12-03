from utils.tool.SentenceClassify import SentenceClassify

sentence_classify = SentenceClassify("ToolModel/sentence_judge_bert03")
print(sentence_classify.classify("这个是左上那个,啊不对,这是右上那个"))
