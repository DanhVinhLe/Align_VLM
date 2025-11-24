#!/bin/bash
set -e

ROOT="finetune_data"
mkdir -p "$ROOT"

# REQUIREMENTS:
# sudo apt-get install -y aria2 parallel unzip

# List of URL + target directory pairs
DATASETS=(
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip|$ROOT/textvqa"
    "https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip|$ROOT/iconqa_data"
    "http://images.cocodataset.org/zips/train2017.zip|$ROOT/coco"
    "https://huggingface.co/datasets/DVLe/sharegpt/resolve/main/sam_images_share-sft.zip|$ROOT/sam"
    "https://huggingface.co/datasets/DVLe/sharegpt/resolve/main/share_textvqa.zip|$ROOT/share_textvqa"
    "https://huggingface.co/datasets/DVLe/sharegpt/resolve/main/web-celebrity.zip|$ROOT/web-celebrity"
    "https://huggingface.co/datasets/DVLe/sharegpt/resolve/main/web-landmark.zip|$ROOT/web-landmark"
    "https://huggingface.co/datasets/DVLe/sharegpt/resolve/main/wikiart.zip|$ROOT/wikiart"
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip|$ROOT/gqa"
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip|$ROOT/vg/VG_100K"
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip|$ROOT/vg/VG_100K_2"
)

echo "========= START MULTI-THREAD DOWNLOAD ========="

download_and_extract() {
    entry="$1"
    url="${entry%%|*}"
    outdir="${entry##*|}"
    zipfile="${outdir}/file.zip"
    
    mkdir -p "$outdir"
    
    # Skip if already has files
    if [ "$(ls -A "$outdir" 2>/dev/null)" ]; then
        echo "[✔] $outdir already exists → skip"
        return
    fi
    
    echo "[↓] Downloading $url → $zipfile"
    aria2c -x 16 -s 16 -o "$zipfile" "$url" --allow-overwrite=true
    
    echo "[*] Unzipping $zipfile ..."
    
    # Unzip vào thư mục tạm
    tmpdir="${outdir}_tmp"
    mkdir -p "$tmpdir"
    unzip -q "$zipfile" -d "$tmpdir"
    
    # Xử lý từng dataset theo cấu trúc riêng
    basename_dir=$(basename "$outdir")
    
    case "$basename_dir" in
        "iconqa_data")
            # iconqa_data/iconqa_data/iconqa → iconqa_data/iconqa
            if [ -d "$tmpdir/iconqa_data/iconqa" ]; then
                mv "$tmpdir/iconqa_data/iconqa" "$outdir/"
            elif [ -d "$tmpdir/iconqa" ]; then
                mv "$tmpdir/iconqa" "$outdir/"
            else
                mv "$tmpdir"/* "$outdir/"
            fi
            ;;
            
        "sam")
            # sam/{images} → sam/images/{images}
            mkdir -p "$outdir/images"
            mv "$tmpdir"/* "$outdir/images/" 2>/dev/null || true
            ;;
            
        "VG_100K"|"VG_100K_2")
            # vg/VG_100K/VG_100K/{images} → vg/VG_100K/{images}
            if [ -d "$tmpdir/$basename_dir" ]; then
                mv "$tmpdir/$basename_dir"/* "$outdir/"
            else
                mv "$tmpdir"/* "$outdir/"
            fi
            ;;
            
        "share_textvqa"|"web-celebrity"|"web-landmark"|"wikiart")
            # wikiart/data/wikiart/images → wikiart/images
            if [ -d "$tmpdir/data/$basename_dir/images" ]; then
                mkdir -p "$outdir/images"
                mv "$tmpdir/data/$basename_dir/images"/* "$outdir/images/"
            elif [ -d "$tmpdir/$basename_dir/images" ]; then
                mv "$tmpdir/$basename_dir/images" "$outdir/"
            elif [ -d "$tmpdir/images" ]; then
                mv "$tmpdir/images" "$outdir/"
            else
                mv "$tmpdir"/* "$outdir/"
            fi
            ;;
            
        *)
            # Default: move tất cả nội dung
            mv "$tmpdir"/* "$outdir/"
            ;;
    esac
    
    # Dọn dẹp
    rm -rf "$tmpdir"
    echo "[🗑] Removing zip: $zipfile"
    rm "$zipfile"
    echo "[✔] Done: $outdir"
}

export -f download_and_extract
export ROOT

# Run parallel downloads (max 4 jobs at once)
printf "%s\n" "${DATASETS[@]}" | parallel -j 4 download_and_extract {}

echo "========= ALL FINISHED ========="

echo "========= SCIENCEQA ========="

SCI_ROOT="$ROOT/ScienceQA"
mkdir -p "$SCI_ROOT"
cd "$SCI_ROOT"

if [ -d "train" ]; then
    echo "[✔] ScienceQA train exists → skip"
else
    wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip -O train.zip
    unzip -q train.zip
    rm train.zip
    echo "[✔] ScienceQA done"
fi

echo "========= ALL FINISHED ========="
