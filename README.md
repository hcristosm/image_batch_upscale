# 📸 Real-ESRGAN Batch Upscaler (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hcristosm/image_batch_upscale/blob/main/batch_upscale.ipynb)

Um pipeline prático e 100% automatizado em Python para aumentar em **4x a resolução** de lotes de imagens utilizando Inteligência Artificial (Real-ESRGAN + GFPGAN), diretamente pelo Google Colab.

---

## ⚡ Recursos Principais

- 📦 **Processamento em Lote:** Aceita fotos avulsas ou arquivos `.zip` (descompacta subpastas automaticamente).
- 👤 **Restauração Facial:** Aplica o algoritmo GFPGAN para reconstruir rostos com nitidez.
- 🛡️ **Anti-Crash & Tratamento de Erros:**
  - Converte imagens Grayscale (preto e branco) para RGB automaticamente para evitar travamentos.
  - Processa por blocos (`--tile 512`), garantindo que fotos gigantes não estourem a memória da GPU.
- 📥 **Download Automático:** Gera e baixa o arquivo `fotos_upscaled.zip` ao final do processo.

---

## 📖 Tutorial: Como Usar

### Passo 1: Abrir no Google Colab
Clique no botão **Open in Colab** no topo deste repositório ou abra o arquivo `.ipynb` direto no Colab.

### Passo 2: Ativar a GPU Gratuita
Antes de rodar o script, certifique-se de que o Colab está usando aceleração por placa de vídeo:
1. No menu superior, vá em **Ambiente de execução** > **Alterar o tipo de ambiente de execução**.
2. Em *Acelerador de hardware*, selecione **GPU T4**.
3. Clique em **Salvar**.

### Passo 3: Executar o Código
1. Clique no botão de **Play (▶)** na célula de código principal (ou pressione `Ctrl + Enter`).
2. Aguarde alguns segundos enquanto o script instala as dependências e aplica as correções de ambiente.

### Passo 4: Enviar suas fotos
1. Quando o botão **"Escolher arquivos"** aparecer na tela, selecione:
   - Uma ou mais fotos avulsas (`.jpg`, `.png`, etc.); **OU**
   - Um arquivo `.zip` contendo todas as suas imagens.
2. O upload começará automaticamente.

### Passo 5: Download dos Resultados
Após o término do processamento, o script irá compactar todas as imagens tratadas e o download do arquivo `fotos_upscaled.zip` iniciará **automaticamente no seu navegador**.

---

## 🛠️ Tecnologias Utilizadas

- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN):** Algoritmo principal de super-resolução.
- **[GFPGAN](https://github.com/TencentARC/GFPGAN):** Módulo de restauração de feições faciais.
- **PyTorch & Torchvision:** Framework de aprendizado de máquina.
- **PIL (Pillow):** Tratamento e conversão de formatos de imagem.

---

## 📜 Licença

Este projeto está sob a licença [MIT](LICENSE). Sinta-se livre para usar, modificar e distribuir.
