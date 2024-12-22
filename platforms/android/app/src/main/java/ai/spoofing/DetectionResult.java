package ai.spoofing;

import android.graphics.Bitmap;

public class DetectionResult {
    public Bitmap faceBitmap; // Changed from Rect to Bitmap
    public float[] probs;

    public DetectionResult(Bitmap faceBitmap, float[] probs) {
        this.faceBitmap = faceBitmap;
        this.probs = probs;
    }
}
