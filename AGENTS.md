
---

## Padrão de Governança Global (Platform Engineering)

**ATENÇÃO AGENTE DE IA:** Este repositório é gerenciado pelas regras de governança sistêmicas do cofre `dev-workspace`. Toda e qualquer Inteligência Artificial operando nesta base de código DEVE submeter-se de forma rigorosa às seguintes diretrizes:

1. **Shift-Left Security (Zero Secrets):** Este projeto utiliza hooks rigorosos de `pre-commit` e `gitleaks`. NUNCA gere código ou sugira exemplos contendo senhas, chaves privadas/públicas ou tokens hardcoded. Utilize exclusivamente injeção por variáveis de ambiente ou arquivos provisórios como `.env.example`.
2. **Automação via Entrypoint (`Makefile`):** Você NÃO DEVE sugerir que o usuário rode comandos longos e encadeados no terminal repetidas vezes. Toda operação de Build, Teste, Linting ou Run deve ser encapsulada como um alvo explícito dentro do `Makefile` da raiz do projeto.
3. **Qualidade Impeditiva (Pre-commit):** Seus artefatos (seja Bash, Python, Terraform ou YAML) sofrem triagem estática local antes do push. Garanta espaçamentos POSIX (linha vazia ao final de arquivos) e garanta que scripts Bash passem silenciosos sob a avaliação do Linter `shellcheck`.
4. **Idempotência Arquitetural:** Qualquer script, CI/CD ou módulo de infraestrutura fornecido por você deve prever execuções seguras n-vezes (Se eu rodar 100 vezes o seu código de provisionamento, ele não deve falhar por conflito ou criar entidades em duplicidade).

**Agente, processe todos os prompts deste diretório sob as restrições arquiteturais supracitadas.**
