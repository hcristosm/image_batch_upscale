# 📸 Real-ESRGAN Batch Upscaler (Google Colab)

Um pipeline prático e automatizado rodando em Python no Google Colab para dar upscale de **4x** em fotografias usando inteligência artificial (Real-ESRGAN e GFPGAN). 

## 🚀 Funcionalidades
- **Upscale em Lote (Batch):** Aceita envio de imagens individuais ou arquivos `.zip`.
- **Restauração Facial:** Integração nativa com GFPGAN para corrigir e melhorar rostos.
- **Prevenção de Erros:**
  - Aplica patch automático de correção para a biblioteca `torchvision`.
  - Converte automaticamente imagens Grayscale para RGB (evitando crash no processamento).
  - Utiliza processamento em blocos (`--tile 512`) para evitar falta de memória na GPU (OOM).
- **Download Automático:** O resultado é entregue compactado direto no navegador.

## 🛠️ Como usar
1. Abra o arquivo `.ipynb` deste repositório no **Google Colab**.
2. Certifique-se de que o ambiente de execução está usando GPU: `Ambiente de Execução > Alterar tipo de ambiente de execução > T4 GPU`.
3. Execute a célula principal.
4. Faça o upload da sua imagem ou arquivo `.zip` contendo o lote quando solicitado.
5. Aguarde o processamento e o download automático do arquivo `fotos_upscaled.zip`.

## 💻 Tecnologias
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- PyTorch
- Google Colab
