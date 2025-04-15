from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import uuid
import json
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import MCSpeechToText.Transcript.Serialization.TranscriptDocument as FBSerializeTranscriptDocument

# Define the Type and Tags enum values manually
class WordType:
    kWord = 0
    kPunctuation = 1
    kEmpty = 2
    kSilence = 3  # Pauses

class WordTags:
    kDisfluency = 0  # Filler words
    kProfanity = 1
    kHead = 2
    kTail = 3
    kNumTags = 4

TicksPerSecond = 254016000000

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return obj.hex
        return json.JSONEncoder.default(self, obj)

def main(args):
    options = parse_args(args[1:])
    if options.input:
        if os.path.isfile(options.input):
            writeJSONFormat(options.input)
        elif os.path.isdir(options.input):
            for file in os.listdir(options.input):
                if file.endswith(".prtranscript"):
                    writeJSONFormat(os.path.join(options.input, file))

def writeJSONFormat(filename):
    index_of_dot = filename.index('.')
    file_name_without_extension = filename[:index_of_dot]

    reference_speakers, ref_segments, ref_words, ref_sentences = readFlatbuffer(filename)

    print("Writing reference words to disk as " + file_name_without_extension + '.words.json')
    with open(file_name_without_extension + '.words.json', 'w') as outfile:
        json.dump(ref_words, outfile, indent=4, cls=UUIDEncoder)
    print("Writing reference sentences to disk as " + file_name_without_extension + '.sentences.json')
    with open(file_name_without_extension + '.sentences.json', 'w') as outfile:
        json.dump(ref_sentences, outfile, indent=4, cls=UUIDEncoder)
    print("Writing reference segments to disk as " + file_name_without_extension + '.segments.json')
    with open(file_name_without_extension + '.segments.json', 'w') as outfile:
        json.dump(ref_segments, outfile, indent=4, cls=UUIDEncoder)

def readFlatbuffer(fb_file):
    buf = bytearray(open(fb_file, 'rb').read())
    data = FBSerializeTranscriptDocument.TranscriptDocument.GetRootAsTranscriptDocument(buf, 0)
    speakers, segments, words, sentences = extractSpeakersWordsAndSegments(data)
    return speakers, segments, words, sentences  

def extractSpeakersWordsAndSegments(transDoc):
    transData = transDoc.TranscriptData()
    numSpeakers = transData.SpeakersLength()
    numSegments = transData.TranscriptSegmentsLength()

    speakers = []
    segments = []
    words = []
    sentences = []

    # First extract speakers
    for i in range(0, numSpeakers):
        speaker = transData.Speakers(i)
        guid = uuid.UUID(int = speaker.SpeakerID().MValue1() << 64 | speaker.SpeakerID().MValue2())
        speaker_obj = {'name': speaker.Name().decode("utf-8"), 'guid': guid, 'segments':[]}
        speakers.append(speaker_obj)

    # Then process segments and words
    for j in range(0, numSegments):
        segment = transData.TranscriptSegments(j)
        segment_speaker_guid = uuid.UUID(int = segment.SpeakerID().MValue1() << 64 | segment.SpeakerID().MValue2())
        speaker_name = next((sp['name'] for sp in speakers if sp['guid'] == segment_speaker_guid), "Unknown")

        # Process words in this segment
        numWords = segment.SegmentWordsLength()
        current_sentence = []
        sentence_start = None
        sentence_end = None
        segWords = []

        for k in range(0, numWords):
            word = segment.SegmentWords(k)
            word_text = word.Word().decode("utf-8") if word.Word() else ""
            start_time = word.Start() / TicksPerSecond
            end_time = (word.Start() + word.Duration()) / TicksPerSecond
            duration = word.Duration() / TicksPerSecond
            
            # Get word type and tags
            word_type = word.Type()
            word_tags = word.Tags()
            
            # Determine if it's a pause or filler
            is_pause = (word_type == WordType.kSilence)
            is_filler = (word_tags & (1 << WordTags.kDisfluency)) != 0
            
            # Create word object for words.json
            word_obj = {
                'speaker_guid': segment_speaker_guid,
                'speaker_name': speaker_name,
                'word': word_text,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'duration': round(duration, 2),
                'isEOS': word.IsEOS(),
                'type': word_type,
                'tags': word_tags,
                'is_pause': is_pause,
                'is_filler': is_filler
            }
            words.append(word_obj)
            segWords.append(word_obj)

            # Track sentence start with first non-empty word
            if not current_sentence and word_text:
                sentence_start = start_time
            
            # Build sentence text with special markers
            if is_pause:
                current_sentence.append("<pause>")
            elif is_filler:
                current_sentence.append(f"<filler>{word_text}</filler>" if word_text else "<filler>")
            elif word_text:  # Regular word
                current_sentence.append(word_text)
            
            # Update sentence end time for every word
            sentence_end = end_time

            # Finalize sentence if EOS marker found
            if word.IsEOS() and current_sentence and sentence_start is not None:
                sentence_text = ' '.join(current_sentence)
                sentences.append({
                    'speaker_guid': segment_speaker_guid,
                    'speaker_name': speaker_name,
                    'sentence': sentence_text,
                    'start': round(sentence_start, 2),
                    'end': round(sentence_end, 2),
                    'duration': round(sentence_end - sentence_start, 2)
                })
                current_sentence = []
                sentence_start = None

        # Handle any remaining words in segment without EOS marker
        if current_sentence and sentence_start is not None:
            sentence_text = ' '.join(current_sentence)
            sentences.append({
                'speaker_guid': segment_speaker_guid,
                'speaker_name': speaker_name,
                'sentence': sentence_text,
                'start': round(sentence_start, 2),
                'end': round(sentence_end, 2),
                'duration': round(sentence_end - sentence_start, 2)
            })

        # Add segment with all words
        segments.append({
            'speaker_guid': segment_speaker_guid,
            'speaker_name': speaker_name,
            'start': segment.Start() / TicksPerSecond,
            'end': (segment.Start() + segment.Duration()) / TicksPerSecond,
            'duration': segment.Duration() / TicksPerSecond,
            'words': segWords
        })

    return speakers, segments, words, sentences

def parse_args(args):
    parser = ArgumentParser(
        description='Plot diarization results',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-r', '--input', type=str, required=True,
                        help="Path to reference prtranscript file.")
    return parser.parse_args(args)

if __name__ == '__main__':
    main(sys.argv)