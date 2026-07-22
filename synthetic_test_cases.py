"""
Synthetic test cases for RAGNAROK evaluation
Covers: multilingual, multi-document, conflicting context, implicit questions,
        hallucinations, omissions, redundancy, retrieval failures, business edge cases.
"""

SYNTHETIC_CASES = [
    # === BASIC CASES (1-5) ===
    {
        "name": "perfect_rag",
        "query": "Когда был построен Эйфелева башня?",
        "contexts": [
            "Эйфелева башня была построена в 1889 году.",
            "Строительство длилось 2 года, с 1887 по 1889."
        ],
        "answer": "Эйфелева башня была построена в 1889 году.",
        "relevance_scores": [1, 1],
        "latency_ms": 150,
        "input_tokens": 15,
        "output_tokens": 10,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "expected_sfc": 0.5,
        "expected_faithfulness": 1.0,
        "description": "Идеальный RAG: полный, точный, релевантный ответ"
    },
    {
        "name": "hallucination_severe",
        "query": "Где находится Эйфелева башня?",
        "contexts": ["Эйфелева башня находится в Париже, Франция."],
        "answer": "Эйфелева башня находится в Лондоне, Великобритания.",
        "relevance_scores": [1],
        "latency_ms": 200,
        "input_tokens": 8,
        "output_tokens": 9,
        "num_chunks_retrieved": 1,
        "has_citations": False,
        "expected_fault": "generation",
        "expected_faithfulness": 0.0,
        "description": "Серьёзная галлюцинация: полная противоположность фактам"
    },
    {
        "name": "incomplete_coverage",
        "query": "Расскажи об Эйфелевой башне",
        "contexts": [
            "Эйфелева башня построена в 1889.",
            "Высота башни 330 метров.",
            "Автор — Гюстав Эйфель.",
            "Башня весит 10100 тонн."
        ],
        "answer": "Эйфелева башня была построена в 1889 году.",
        "relevance_scores": [1, 1, 1, 1],
        "latency_ms": 180,
        "input_tokens": 12,
        "output_tokens": 10,
        "num_chunks_retrieved": 4,
        "has_citations": False,
        "expected_fault": "chunking",
        "expected_sfc": 0.25,
        "description": "Неполное покрытие: ответ упускает 3 из 4 тем"
    },
    {
        "name": "verbose_redundant",
        "query": "Когда построили Эйфелеву башню?",
        "contexts": ["Эйфелева башня была построена в 1889 году."],
        "answer": "Эйфелева башня была построена в 1889 году. Да, именно в 1889 году. В 1889 году она была построена. Это произошло в 1889.",
        "relevance_scores": [1],
        "latency_ms": 250,
        "input_tokens": 8,
        "output_tokens": 28,
        "num_chunks_retrieved": 1,
        "has_citations": False,
        "expected_fault": "chunking",
        "expected_ids": 0.05,
        "description": "Избыточность: много слов, мало смысла"
    },
    {
        "name": "retrieval_failure",
        "query": "Кто изобрёл телефон?",
        "contexts": [],
        "answer": "Телефон изобрёл Александр Белл в 1876 году.",
        "relevance_scores": [],
        "latency_ms": 300,
        "input_tokens": 6,
        "output_tokens": 10,
        "num_chunks_retrieved": 0,
        "has_citations": False,
        "expected_fault": "out_of_scope",
        "description": "Провал ретривала: нет контекста, но ответ дан"
    },

    # === MULTILINGUAL CASES (6-10) ===
    {
        "name": "multilingual_en_ru",
        "query": "What is the capital of France?",
        "contexts": [
            "Париж — столица Франции.",
            "В Париже находится Эйфелева башня."
        ],
        "answer": "The capital of France is Paris.",
        "relevance_scores": [1, 1],
        "latency_ms": 200,
        "input_tokens": 12,
        "output_tokens": 8,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Мультиязычный: запрос на английском, контекст на русском"
    },
    {
        "name": "multilingual_ru_en",
        "query": "Что такое квантовое запутывание?",
        "contexts": [
            "Quantum entanglement is a physical phenomenon.",
            "When particles interact, their quantum states become correlated."
        ],
        "answer": "Квантовое запутывание — это физическое явление, при котором квантовые состояния частиц коррелируют.",
        "relevance_scores": [1, 1],
        "latency_ms": 220,
        "input_tokens": 10,
        "output_tokens": 16,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Мультиязычный: запрос на русском, контекст на английском"
    },
    {
        "name": "multilingual_de_ru",
        "query": "Wer hat die Relativitätstheorie entwickelt?",
        "contexts": [
            "Альберт Эйнштейн разработал теорию относительности в 1905 году.",
            "Специальная теория относительности изменила понимание пространства и времени."
        ],
        "answer": "Теорию относительности разработал Альберт Эйнштейн.",
        "relevance_scores": [1, 1],
        "latency_ms": 190,
        "input_tokens": 14,
        "output_tokens": 10,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Мультиязычный: запрос на немецком, контекст на русском"
    },
    {
        "name": "multilingual_mixed_hallucination",
        "query": "Capital of Japan?",
        "contexts": ["Столица Японии — Токио.", "Tokyo is the largest city in Japan."],
        "answer": "The capital of Japan is Kyoto.",
        "relevance_scores": [1, 1],
        "latency_ms": 180,
        "input_tokens": 8,
        "output_tokens": 8,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "generation",
        "expected_faithfulness": 0.0,
        "description": "Мультиязычный + галлюцинация: правильный контекст, но неверный ответ"
    },
    {
        "name": "multilingual_fr_ru",
        "query": "Quelle est la hauteur de la Tour Eiffel?",
        "contexts": ["Высота Эйфелевой башни составляет 330 метров."],
        "answer": "Высота Эйфелевой башни — 330 метров.",
        "relevance_scores": [1],
        "latency_ms": 170,
        "input_tokens": 12,
        "output_tokens": 8,
        "num_chunks_retrieved": 1,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Мультиязычный: запрос на французском, контекст на русском"
    },

    # === MULTI-DOCUMENT CASES (11-15) ===
    {
        "name": "multi_doc_complete",
        "query": "Какие достопримечательности есть в Париже?",
        "contexts": [
            "Эйфелева башня — символ Парижа, построена в 1889.",
            "Лувр — крупнейший музей мира, бывший королевский дворец.",
            "Нотр-Дам — готический собор на острове Сите.",
            "Триумфальная арка была построена в честь побед Наполеона."
        ],
        "answer": "В Париже есть Эйфелева башня (1889), Лувр, собор Нотр-Дам и Триумфальная арка.",
        "relevance_scores": [1, 1, 1, 1],
        "latency_ms": 250,
        "input_tokens": 12,
        "output_tokens": 20,
        "num_chunks_retrieved": 4,
        "has_citations": False,
        "expected_fault": "healthy",
        "expected_sfc": 1.0,
        "description": "Мультидокументный: полное покрытие 4 источников"
    },
    {
        "name": "multi_doc_partial",
        "query": "Какие достопримечательности есть в Париже?",
        "contexts": [
            "Эйфелева башня — символ Парижа.",
            "Лувр — крупнейший музей мира.",
            "Нотр-Дам — готический собор.",
            "Триумфальная арка — памятник победам."
        ],
        "answer": "В Париже есть Эйфелева башня и Лувр.",
        "relevance_scores": [1, 1, 1, 1],
        "latency_ms": 200,
        "input_tokens": 12,
        "output_tokens": 10,
        "num_chunks_retrieved": 4,
        "has_citations": False,
        "expected_fault": "chunking",
        "expected_sfc": 0.5,
        "description": "Мультидокументный: покрытие только 2 из 4 источников"
    },
    {
        "name": "multi_doc_low_ndcg",
        "query": "Столица Франции",
        "contexts": [
            "Лион — город во Франции.",
            "Париж — столица Франции.",
            "Марсель — портовый город."
        ],
        "answer": "Столица Франции — Париж.",
        "relevance_scores": [0, 1, 0],
        "latency_ms": 180,
        "input_tokens": 8,
        "output_tokens": 6,
        "num_chunks_retrieved": 3,
        "has_citations": False,
        "expected_fault": "retrieval",
        "description": "Мультидокументный: релевантный документ на 2-й позиции (низкий NDCG)"
    },
    {
        "name": "multi_doc_contradiction",
        "query": "Какая высота Эйфелевой башни?",
        "contexts": [
            "Высота Эйфелевой башни — 330 метров.",
            "Эйфелева башня имеет высоту 324 метра до вершины антенны."
        ],
        "answer": "Эйфелева башня имеет высоту 330 метров.",
        "relevance_scores": [1, 1],
        "latency_ms": 190,
        "input_tokens": 10,
        "output_tokens": 8,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Мультидокументный с противоречивыми данными (330 vs 324)"
    },
    {
        "name": "multi_doc_with_citations",
        "query": "Когда открылись первые Олимпийские игры?",
        "contexts": [
            "Первые Олимпийские игры современности прошли в 1896 году в Афинах.",
            "Игры были организованы Международным олимпийским комитетом."
        ],
        "answer": "Первые Олимпийские игры современности прошли в 1896 году [Источник 1].",
        "relevance_scores": [1, 1],
        "latency_ms": 210,
        "input_tokens": 12,
        "output_tokens": 14,
        "num_chunks_retrieved": 2,
        "has_citations": True,
        "expected_fault": "healthy",
        "description": "Мультидокументный с цитированием источников"
    },

    # === CONFLICTING CONTEXT CASES (16-20) ===
    {
        "name": "conflicting_dates",
        "query": "Когда родился Шекспир?",
        "contexts": [
            "Уильям Шекспир родился 23 апреля 1564 года.",
            "По некоторым данным, Шекспир родился 26 апреля 1564 года."
        ],
        "answer": "Уильям Шекспир родился 23 апреля 1564 года.",
        "relevance_scores": [1, 1],
        "latency_ms": 180,
        "input_tokens": 10,
        "output_tokens": 10,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Противоречивый контекст: две даты, но обе правдоподобны"
    },
    {
        "name": "conflicting_opinions",
        "query": "Какой язык программирования лучше?",
        "contexts": [
            "Python — лучший язык для data science и машинного обучения.",
            "Rust превосходит Python по производительности и безопасности памяти."
        ],
        "answer": "Python лучше для data science, а Rust — для системного программирования.",
        "relevance_scores": [1, 1],
        "latency_ms": 220,
        "input_tokens": 12,
        "output_tokens": 14,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "expected_sfc": 1.0,
        "description": "Противоречивые мнения: ответ синтезирует обе позиции"
    },
    {
        "name": "conflicting_facts_wrong",
        "query": "Какая температура кипения воды?",
        "contexts": [
            "Вода кипит при 100°C на уровне моря.",
            "Вода кипит при 90°C на высоте 3000 метров."
        ],
        "answer": "Вода всегда кипит при 100°C.",
        "relevance_scores": [1, 1],
        "latency_ms": 190,
        "input_tokens": 10,
        "output_tokens": 8,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "generation",
        "description": "Противоречивый контекст: игнорирование условий (давление)"
    },
    {
        "name": "conflicting_sources",
        "query": "Какая площадь России?",
        "contexts": [
            "Площадь России составляет 17,1 млн км².",
            "По данным на 2024 год, площадь России — 17,098 млн км²."
        ],
        "answer": "Площадь России — примерно 17,1 млн км².",
        "relevance_scores": [1, 1],
        "latency_ms": 200,
        "input_tokens": 10,
        "output_tokens": 10,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Противоречивые источники: мелкие расхождения, ответ обобщает"
    },
    {
        "name": "conflicting_hallucination",
        "query": "Кто написал 'Войну и мир'?",
        "contexts": [
            "Роман 'Война и мир' написал Лев Толстой.",
            "Некоторые исследователи считают, что часть текста написана Софьей Толстой."
        ],
        "answer": "Роман 'Война и мир' написал Фёдор Достоевский.",
        "relevance_scores": [1, 1],
        "latency_ms": 180,
        "input_tokens": 10,
        "output_tokens": 10,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "generation",
        "expected_faithfulness": 0.0,
        "description": "Противоречивый контекст + галлюцинация: полностью неверный автор"
    },

    # === IMPLICIT QUESTIONS (21-25) ===
    {
        "name": "implicit_causal",
        "query": "Почему небо голубое?",
        "contexts": [
            "Рассеяние Релея — это рассеяние света на частицах, размер которых меньше длины волны.",
            "Голубой цвет неба объясняется рассеянием коротких волн (синего) сильнее, чем длинных (красного)."
        ],
        "answer": "Небо голубое из-за рассеяния Релея: короткие синие волны рассеиваются сильнее длинных.",
        "relevance_scores": [1, 1],
        "latency_ms": 220,
        "input_tokens": 10,
        "output_tokens": 16,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Неявный вопрос 'почему' — требует объяснения причин"
    },
    {
        "name": "implicit_comparison",
        "query": "Python или JavaScript?",
        "contexts": [
            "Python — интерпретируемый язык с динамической типизацией.",
            "JavaScript — язык для веб-разработки, работает в браузере."
        ],
        "answer": "Python лучше для data science и backend, JavaScript — для frontend и веба.",
        "relevance_scores": [1, 1],
        "latency_ms": 210,
        "input_tokens": 8,
        "output_tokens": 14,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Неявное сравнение: запрос требует сравнительного анализа"
    },
    {
        "name": "implicit_how_to",
        "query": "Заварить идеальный чай",
        "contexts": [
            "Для зелёного чая используйте воду 70-80°C, заваривайте 2-3 минуты.",
            "Чёрный чай заваривается водой 90-95°C в течение 3-5 минут."
        ],
        "answer": "Для зелёного чая: 70-80°C, 2-3 минуты. Для чёрного: 90-95°C, 3-5 минут.",
        "relevance_scores": [1, 1],
        "latency_ms": 200,
        "input_tokens": 8,
        "output_tokens": 18,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Неявная инструкция: запрос требует пошагового ответа"
    },
    {
        "name": "implicit_opinion",
        "query": "Стоит ли переходить на удалёнку?",
        "contexts": [
            "Удалённая работа повышает гибкость и снижает затраты на офис.",
            "Удалёнка может снизить командную сплочённость и увеличить выгорание."
        ],
        "answer": "Удалённая работа имеет плюсы (гибкость, экономия) и минусы (изоляция, выгорание).",
        "relevance_scores": [1, 1],
        "latency_ms": 230,
        "input_tokens": 10,
        "output_tokens": 14,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Неявный запрос мнения: требует сбалансированного ответа"
    },
    {
        "name": "implicit_omission",
        "query": "Всё о квантовых компьютерах",
        "contexts": [
            "Кубиты используют суперпозицию для параллельных вычислений.",
            "Квантовая запутанность позволяет коррелировать состояния кубитов.",
            "Декогеренция — главная проблема квантовых систем."
        ],
        "answer": "Квантовые компьютеры используют кубиты.",
        "relevance_scores": [1, 1, 1],
        "latency_ms": 190,
        "input_tokens": 10,
        "output_tokens": 6,
        "num_chunks_retrieved": 3,
        "has_citations": False,
        "expected_fault": "chunking",
        "expected_sfc": 0.33,
        "description": "Неявный запрос 'всё о' — ожидается полнота, но ответ слишком краток"
    },

    # === BUSINESS EDGE CASES (26-30) ===
    {
        "name": "business_expensive",
        "query": "Напиши подробный анализ мировой экономики",
        "contexts": [
            "Мировой ВВП вырос на 3.2% в 2024 году.",
            "Инфляция в развитых странах снизилась до 2.8%."
        ],
        "answer": "Подробный многостраничный анализ мировой экономики с графиками и прогнозами...",
        "relevance_scores": [1, 1],
        "latency_ms": 5000,
        "input_tokens": 20,
        "output_tokens": 2000,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "business",
        "description": "Бизнес-кейс: слишком дорого и медленно для простого запроса"
    },
    {
        "name": "business_fast_cheap",
        "query": "2+2",
        "contexts": ["2 + 2 = 4"],
        "answer": "4",
        "relevance_scores": [1],
        "latency_ms": 50,
        "input_tokens": 3,
        "output_tokens": 1,
        "num_chunks_retrieved": 1,
        "has_citations": False,
        "expected_fault": "healthy",
        "description": "Бизнес-кейс: быстро, дёшево, простой запрос — идеальная эффективность"
    },
    {
        "name": "business_low_trust",
        "query": "Как лечить рак?",
        "contexts": [
            "Лечение рака включает химиотерапию, лучевую терапию и иммунотерапию.",
            "Конкретный метод зависит от типа и стадии опухоли."
        ],
        "answer": "Рак лечится травами и медитацией, официальная медицина бесполезна.",
        "relevance_scores": [1, 1],
        "latency_ms": 200,
        "input_tokens": 10,
        "output_tokens": 12,
        "num_chunks_retrieved": 2,
        "has_citations": False,
        "expected_fault": "generation",
        "description": "Бизнес-кейс: критически низкий trust score (медицинская дезинформация)"
    },
    {
        "name": "business_waste",
        "query": "Какая столица Италии?",
        "contexts": [
            "Рим — столица Италии.",
            "Милан — финансовый центр Италии.",
            "Венеция — город на воде в Италии.",
            "Флоренция — культурная столица Италии.",
            "Неаполь — город на юге Италии."
        ],
        "answer": "Столица Италии — Рим.",
        "relevance_scores": [1, 0, 0, 0, 0],
        "latency_ms": 300,
        "input_tokens": 8,
        "output_tokens": 6,
        "num_chunks_retrieved": 5,
        "has_citations": False,
        "expected_fault": "retrieval",
        "description": "Бизнес-кейс: высокий waste ratio (5 чанков, использован 1)"
    },
    {
        "name": "business_sla_breach",
        "query": "Погода в Москве",
        "contexts": ["В Москве сегодня солнечно, +15°C."],
        "answer": "В Москве сегодня солнечно, +15°C.",
        "relevance_scores": [1],
        "latency_ms": 5000,
        "input_tokens": 6,
        "output_tokens": 8,
        "num_chunks_retrieved": 1,
        "has_citations": False,
        "expected_fault": "business",
        "description": "Бизнес-кейс: правильный ответ, но нарушение SLA (5 секунд для простого запроса)"
    },
]
