from django import forms

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(
        label="Select Video",
        required=True,
        widget=forms.FileInput(attrs={"accept": "video/*"})
    )
    sequence_length = forms.IntegerField(
        label="Sequence Length",
        required=True,
        min_value=1,
        max_value=300,  # Example upper limit
        initial=30
    )

    def clean_upload_video_file(self):
        file = self.cleaned_data.get("upload_video_file")
        if file:
            if file.size > 50 * 1024 * 1024:  # 50 MB limit
                raise forms.ValidationError("Video file too large (max 50 MB).")
            if not file.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                raise forms.ValidationError("Unsupported video format.")
        return file