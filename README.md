# ComfyUI JoyCaption (fork)

更新日: 2026-09-12

## 概要

画像からキャプションを生成する JoyCaption GGUF ノードです。ComfyUI-JoyCaptionのfork版です。キャプション種別・長さ・追加オプションは上流 [ComfyUI-JoyCaption](https://github.com/1038lab/ComfyUI-JoyCaption) と同じで、推論は `llama-cli --mmproj --image` で行います。llama-cpp-python は使いません。transformers の HF ノードは含みません。上流本体は変更しません。

[ComfyUI-llama-cli](../ComfyUI-llama-cli) が必要です。未導入なら `Install custom_nodes/ComfyUI-llama-cli and restart ComfyUI.` で止まります。

llama-cli は毎回プロセスを終了するので VRAM は空きます。`Keep in Memory` / `Global Cache` は GGUF を DRAM にピンして次回のロードを速くします。GPU 上には残しません。キャプション文言は上流の `Llava15ChatHandler` と一致しません。

## ノード

- `JoyCaption GGUF (fork)`
- `JoyCaption GGUF Advanced (fork)`
- `JoyCaption Extra Options (fork)`

カテゴリ: `JoyCaption-fork`。上流の `JC_GGUF` とは別名なので同時インストールできます。

モデルは `ComfyUI/models/LLM/GGUF`。無ければ Hugging Face から取得します。mmproj は上流と同じ `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`。

## memory_management

- `Clear After Run`: DRAM ピンもしない
- `Keep in Memory` / `Global Cache`: GGUF を DRAM にピンして次回のロードを速くする。GPU 上には残さない
