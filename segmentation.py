from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines import OverlappedSpeechDetection

model = Model.from_pretrained(
  "pyannote/segmentation-3.0", 
  use_auth_token="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")

pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("facebookdenoiser.wav")

print ("===============SPEACH REGIONS=========")
print (vad)


pipeline = OverlappedSpeechDetection(segmentation=model)
HYPER_PARAMETERS = {
  # remove overlapped speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-overlapped speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline("facebookdenoiser.wav")


print ("===============Overlapped speech detection=========")
print (osd)