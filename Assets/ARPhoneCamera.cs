using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ARPhoneCamera : MonoBehaviour
{
    [Header("AR")]
    [Tooltip("ARCameraManager on your AR Camera")]
    public ARCameraManager cameraManager;

    [Header("Detector")]
    [Tooltip("Reference to YoloDetectorV8 component with model+labels configured")]
    public YoloDetectorV8 detector;   // <-- fixed type name

    [Header("Perf")]
    [Tooltip("Cap the detector updates to this FPS. 0 = run every frame")]
    [Range(0, 60)] public int targetDetectionsPerSecond = 15;

    [Header("UI Overlay (optional)")]
    [Tooltip("Parent Canvas (Screen Space - Overlay recommended)")]
    public Canvas overlayCanvas;
    [Tooltip("RectTransform covering the screen that will host boxes (usually a full-screen panel)")]
    public RectTransform overlayRoot;
    [Tooltip("Color for box outlines")]
    public Color boxColor = new Color(0f, 1f, 0f, 0.95f);
    [Tooltip("Width of box outline in pixels")]
    public float boxThickness = 3f;
    [Tooltip("Text prefab (optional). If null, simple labels are created at runtime.")]
    public Text labelPrefab;

    // Use YoloDetection here (was Detection)
    public event Action<List<YoloDetection>, Texture2D> OnDetections;

    private Texture2D _cameraTex;
    private List<YoloDetection> _latest = new List<YoloDetection>();
    private float _nextRunTime;
    private BoxPool _pool;

    void Awake()
    {
        if (overlayCanvas != null && overlayRoot != null)
            _pool = new BoxPool(overlayRoot, boxColor, boxThickness, labelPrefab);
    }

    void OnEnable()
    {
        if (cameraManager != null)
            cameraManager.frameReceived += OnCameraFrameReceived;
    }

    void OnDisable()
    {
        if (cameraManager != null)
            cameraManager.frameReceived -= OnCameraFrameReceived;
    }

    void OnDestroy()
    {
        if (_cameraTex != null) Destroy(_cameraTex);
        if (_pool != null) _pool.Dispose();
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs _)
    {
        if (cameraManager == null || detector == null) return;

        // FPS cap for detection
        if (targetDetectionsPerSecond > 0 && Time.time < _nextRunTime)
            return;

        if (!cameraManager.TryAcquireLatestCpuImage(out var image))
            return;

        using (image)
        {
            const TextureFormat fmt = TextureFormat.RGBA32;

            if (_cameraTex == null ||
                _cameraTex.width != image.width ||
                _cameraTex.height != image.height ||
                _cameraTex.format != fmt)
            {
                if (_cameraTex != null) Destroy(_cameraTex);
                _cameraTex = new Texture2D(image.width, image.height, fmt, false);
            }

            var conv = new XRCpuImage.ConversionParams(image, fmt, XRCpuImage.Transformation.MirrorY);
            var raw = _cameraTex.GetRawTextureData<byte>();
            image.Convert(conv, raw);
            _cameraTex.Apply(false, false);
        }

        // Run detector
        _latest = detector.Run(_cameraTex);

        // Notify listeners
        OnDetections?.Invoke(_latest, _cameraTex);

        // Draw overlay (UI)
        if (_pool != null)
            DrawOverlayUI(_latest, _cameraTex);

        if (targetDetectionsPerSecond > 0)
            _nextRunTime = Time.time + 1f / Mathf.Max(1, targetDetectionsPerSecond);
    }

    void DrawOverlayUI(List<YoloDetection> dets, Texture2D camTex)
    {
        if (overlayRoot == null || camTex == null) return;

        var rootRect = overlayRoot.rect;
        float rootW = rootRect.width;
        float rootH = rootRect.height;

        float scale = rootW / camTex.width;
        float drawH = camTex.height * scale;

        Vector2 originTL = new Vector2(-rootW * 0.5f, rootH * 0.5f); // local top-left

        _pool.Begin();

        for (int i = 0; i < dets.Count; i++)
        {
            var d = dets[i];

            float x = d.rect.x * scale;
            float yFromBottom = d.rect.y * scale;
            float w = d.rect.width * scale;
            float h = d.rect.height * scale;

            float tlx = originTL.x + x;
            float tly = originTL.y - (yFromBottom + h);

            _pool.DrawBox(new Rect(tlx, tly, w, h), d.label, d.score);
        }

        _pool.End();
    }

    // ======= Simple UI box pool (outline + label) =======
    class BoxPool : IDisposable
    {
        struct Item
        {
            public RectTransform root;
            public Image top, left, right, bottom;
            public Text label;
        }

        readonly RectTransform parent;
        readonly Color color;
        readonly float thickness;
        readonly Text labelPrefab;

        readonly List<Item> items = new List<Item>();
        int used;

        public BoxPool(RectTransform parent, Color color, float thickness, Text labelPrefab)
        {
            this.parent = parent;
            this.color = color;
            this.thickness = Mathf.Max(1f, thickness);
            this.labelPrefab = labelPrefab;
        }

        public void Begin() { used = 0; }

        public void DrawBox(Rect rectTL, string label, float score)
        {
            var it = Get();
            var root = it.root;

            root.anchorMin = new Vector2(0f, 1f);
            root.anchorMax = new Vector2(0f, 1f);
            root.pivot = new Vector2(0f, 1f);
            root.anchoredPosition = new Vector2(rectTL.x, rectTL.y);
            root.sizeDelta = new Vector2(rectTL.width, rectTL.height);

            LayoutEdge(it.top,    0, 0, rectTL.width, thickness);
            LayoutEdge(it.left,   0, 0, thickness, rectTL.height);
            LayoutEdge(it.right,  rectTL.width - thickness, 0, thickness, rectTL.height);
            LayoutEdge(it.bottom, 0, rectTL.height - thickness, rectTL.width, thickness);

            it.top.color = color; it.left.color = color; it.right.color = color; it.bottom.color = color;

            if (it.label != null)
            {
                string text = string.IsNullOrEmpty(label) ? $"{score:0.00}" : $"{label} {score:0.00}";
                it.label.text = text;
                var lr = it.label.rectTransform;
                lr.anchorMin = new Vector2(0f, 1f);
                lr.anchorMax = new Vector2(0f, 1f);
                lr.pivot = new Vector2(0f, 0f);
                lr.anchoredPosition = new Vector2(0f, -lr.sizeDelta.y - 2f);
                it.label.color = Color.white;
                it.label.gameObject.SetActive(true);
            }
        }

        public void End()
        {
            for (int i = used; i < items.Count; i++)
                items[i].root.gameObject.SetActive(false);
        }

        public void Dispose()
        {
            for (int i = 0; i < items.Count; i++)
                if (items[i].root != null)
                    UnityEngine.Object.Destroy(items[i].root.gameObject);
            items.Clear();
        }

        Item Get()
        {
            if (used < items.Count)
            {
                var it = items[used++];
                it.root.gameObject.SetActive(true);
                return it;
            }

            var go = new GameObject("box", typeof(RectTransform));
            var root = go.GetComponent<RectTransform>();
            root.SetParent(parent, false);
            root.gameObject.SetActive(true);

            var top = MakeEdge("top", root);
            var left = MakeEdge("left", root);
            var right = MakeEdge("right", root);
            var bottom = MakeEdge("bottom", root);

            Text lab = null;
            if (labelPrefab != null)
            {
                lab = UnityEngine.Object.Instantiate<Text>(labelPrefab, root);
            }
            else
            {
                var lgo = new GameObject("label", typeof(RectTransform));
                var rt = lgo.GetComponent<RectTransform>();
                rt.SetParent(root, false);
                var t = lgo.AddComponent<Text>();
                t.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
                t.fontSize = 18;
                t.alignment = TextAnchor.UpperLeft;
                t.horizontalOverflow = HorizontalWrapMode.Overflow;
                t.verticalOverflow = VerticalWrapMode.Overflow;
                lab = t;
            }

            var item = new Item { root = root, top = top, left = left, right = right, bottom = bottom, label = lab };
            items.Add(item);
            used++;
            return item;
        }

        static Image MakeEdge(string name, RectTransform parent)
        {
            var go = new GameObject(name, typeof(RectTransform));
            var rt = go.GetComponent<RectTransform>();
            rt.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.raycastTarget = false;
            return img;
        }

        static void LayoutEdge(Image img, float x, float y, float w, float h)
        {
            var rt = img.rectTransform;
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.anchoredPosition = new Vector2(x, y);
            rt.sizeDelta = new Vector2(w, h);
        }
    }
}
