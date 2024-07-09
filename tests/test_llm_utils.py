import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils import RAG, get_llm_response


@pytest.fixture
def rag_instance():
    with patch('utils.llm_utils.SentenceTransformer'), \
            patch('utils.llm_utils.faiss'):
        yield RAG('test_path', 'test_model', 5, 0.7)


def test_rag_initialization(rag_instance):
    assert rag_instance.max_retrieved_documents == 5
    assert rag_instance.similarity_threshold == 0.7


def test_rag_retrieve(rag_instance):
    rag_instance.knowledge_base = [{'content': 'Test content'}]
    rag_instance.index.search = MagicMock(return_value=([0.5], [[0]]))

    result = rag_instance.retrieve('test query')

    assert len(result) == 1
    assert result[0]['content'] == 'Test content'


@pytest.mark.asyncio
async def test_get_llm_response_openai(rag_instance):
    config = {
        'llm': {
            'provider': 'openai',
            'models': {
                'default': {
                    'name': 'gpt-3',
                    'api_key': 'test_key',
                    'max_tokens': 100,
                    'temperature': 0.7
                }
            },
            'use_fine_tuned': False
        }
    }

    with patch('utils.llm_utils.openai.ChatCompletion.acreate', new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices[0].message.content = 'Test response'

        result = await get_llm_response('Test prompt', config, rag_instance)

        assert result == 'Test response'
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_get_llm_response_anthropic(rag_instance):
    config = {
        'llm': {
            'provider': 'anthropic'
        },
        'alternative_providers': {
            'anthropic': {
                'api_key': 'test_key',
                'model': 'test_model',
                'max_tokens': 100,
                'temperature': 0.7
            }
        }
    }

    with patch('utils.llm_utils.Anthropic', new_callable=AsyncMock) as mock_anthropic:
        mock_anthropic.return_value.completions.create.return_value.completion = 'Test response'

        result = await get_llm_response('Test prompt', config, rag_instance)

        assert result == 'Test response'
        mock_anthropic.assert_called_once()


@pytest.mark.asyncio
async def test_get_llm_response_huggingface(rag_instance):
    config = {
        'llm': {
            'provider': 'huggingface'
        },
        'alternative_providers': {
            'huggingface': {
                'model': 'test_model',
                'max_tokens': 100,
                'temperature': 0.7
            }
        }
    }

    with patch('utils.llm_utils.pipeline', return_value=lambda *args, **kwargs: [{'generated_text': 'Test response'}]):
        result = await get_llm_response('Test prompt', config, rag_instance)

        assert result == 'Test response'


@pytest.mark.asyncio
async def test_get_llm_response_unsupported_provider(rag_instance):
    config = {
        'llm': {
            'provider': 'unsupported'
        }
    }

    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        await get_llm_response('Test prompt', config, rag_instance)


def test_rag_load_knowledge_base(tmp_path):
    kb_file = tmp_path / "test_kb.json"
    kb_file.write_text('[ {"content": "Test content 1"}, {"content": "Test content 2"} ]')

    rag = RAG(str(kb_file), 'test_model', 5, 0.7)

    assert len(rag.knowledge_base) == 2
    assert rag.knowledge_base[0]['content'] == "Test content 1"
    assert rag.knowledge_base[1]['content'] == "Test content 2"


def test_rag_build_index(rag_instance):
    rag_instance.knowledge_base = [{'content': 'Test content 1'}, {'content': 'Test content 2'}]
    rag_instance.sentence_model.encode = MagicMock(return_value=[[1, 2], [3, 4]])

    index = rag_instance.build_index()

    assert index is not None
    rag_instance.sentence_model.encode.assert_called_once_with(['Test content 1', 'Test content 2'])
